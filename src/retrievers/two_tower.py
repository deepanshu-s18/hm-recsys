"""Two-Tower (Dual Encoder) Neural Retrieval Model.

Implements the industrial-standard Two-Tower architecture for large-scale
neural retrieval. Separate user and item encoders produce dense embedding
vectors; items are retrieved via Maximum Inner Product Search (MIPS) using FAISS.

Architecture overview:
    User Tower: user_idx → Embedding → MLP → L2-normalized user vector
    Item Tower: item_idx → Embedding → MLP → L2-normalized item vector
    Score: cosine_similarity(user_vec, item_vec) via inner product

Training objective: InfoNCE (in-batch negative sampling) contrastive loss.
Each batch contains N positives; the remaining N-1 items are treated as
negatives (in-batch negatives). This scales efficiently without explicit
negative sampling.

Design decisions:
    - Separate embedding dims allow different expressiveness for users/items
    - Temperature τ in softmax controls hardness of negative examples
    - L2 normalization enables cosine similarity via efficient FAISS IP search
    - Mixed precision training (bfloat16) where supported for memory efficiency

References:
    Yi et al. "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item
    Recommendations." RecSys 2019. (Google's Two-Tower)

    Karpukhin et al. "Dense Passage Retrieval for Open-Domain QA." EMNLP 2020.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.retrievers.base import BaseRetriever, RetrievalResult
from src.utils.logger import get_logger
from src.utils.timer import timer

log = get_logger(__name__)


class InteractionDataset(Dataset):
    """PyTorch Dataset for user-item interactions.

    Args:
        user_indices: Array of user integer indices.
        item_indices: Array of item integer indices (positive pairs).
    """

    def __init__(self, user_indices: np.ndarray, item_indices: np.ndarray) -> None:
        assert len(user_indices) == len(item_indices)
        self.user_indices = torch.from_numpy(user_indices.astype(np.int64))
        self.item_indices = torch.from_numpy(item_indices.astype(np.int64))

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.user_indices[idx], self.item_indices[idx]


class TowerBlock(nn.Module):
    """Single MLP block for user or item towers.

    Architecture: Embedding → [LayerNorm → Linear → GELU → Dropout] × L → L2Norm

    Args:
        n_entities: Vocabulary size (n_users or n_items).
        embedding_dim: Input embedding dimension.
        hidden_dims: List of hidden layer sizes.
        output_dim: Final embedding dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_entities: int,
        embedding_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Note: Do not set padding_idx=0 as entity 0 is the most popular entity
        self.embedding = nn.Embedding(n_entities, embedding_dim)

        layers: List[nn.Module] = []
        in_dim = embedding_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, h_dim, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim, bias=False))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform init for linear layers, normal for embeddings."""
        nn.init.normal_(self.embedding.weight, std=0.01)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Encode entity indices to L2-normalized embedding.

        Args:
            indices: Long tensor of shape (batch_size,).

        Returns:
            Float tensor of shape (batch_size, output_dim), L2-normalized.
        """
        x = self.embedding(indices)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)


class TwoTowerModel(nn.Module):
    """Dual-encoder model with in-batch negative contrastive loss and Log-Q correction.

    Args:
        n_users: Number of unique users.
        n_items: Number of unique items.
        user_embedding_dim: User tower input embedding size.
        item_embedding_dim: Item tower input embedding size.
        hidden_dims: Shared MLP hidden layer sizes.
        output_dim: Final embedding dimension for both towers.
        dropout: Dropout probability.
        temperature: Softmax temperature τ for InfoNCE loss.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_embedding_dim: int = 64,
        item_embedding_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 128,
        dropout: float = 0.2,
        temperature: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.temperature = temperature

        self.user_tower = TowerBlock(
            n_entities=n_users,
            embedding_dim=user_embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.item_tower = TowerBlock(
            n_entities=n_items,
            embedding_dim=item_embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        item_log_q: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute InfoNCE loss with in-batch negatives and Log-Q sampling debiasing (Yi et al. 2019).

        For a batch of N (user, item) positive pairs:
        - Each user's positive item is at position i on the diagonal
        - All N items serve as negatives for other users
        - Log-Q correction (s(u, i) - log(p_i)) removes popularity sampling bias

        Args:
            user_indices: Long tensor (batch_size,).
            item_indices: Long tensor (batch_size,).
            item_log_q: Optional float tensor (n_items,) with log(p_item)

        Returns:
            Scalar InfoNCE loss.
        """
        user_emb = self.user_tower(user_indices)   # (B, D)
        item_emb = self.item_tower(item_indices)   # (B, D)

        # Scaled cosine similarity matrix (B × B)
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature

        # Apply Log-Q correction to in-batch negative logits if item probabilities provided
        if item_log_q is not None:
            batch_log_q = item_log_q[item_indices].unsqueeze(0)  # (1, B)
            logits = logits - batch_log_q

        # Diagonal entries are positives; all others are in-batch negatives
        labels = torch.arange(len(user_indices), device=user_indices.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def encode_users(self, user_indices: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized user embeddings.

        Args:
            user_indices: Long tensor of user indices.

        Returns:
            Float tensor (n, output_dim).
        """
        return self.user_tower(user_indices)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized item embeddings.

        Args:
            item_indices: Long tensor of item indices.

        Returns:
            Float tensor (n, output_dim).
        """
        return self.item_tower(item_indices)


class TwoTowerRetriever(BaseRetriever):
    """Neural Two-Tower retrieval with FAISS MIPS indexing.

    Full training pipeline including Log-Q sampling bias correction,
    validation loss tracking, and early stopping.

    Args:
        user_dim: User embedding dimension.
        item_dim: Item embedding dimension.
        embedding_dim: Output (shared) embedding dimension.
        hidden_dims: MLP hidden layer sizes.
        dropout: Dropout rate.
        learning_rate: Adam learning rate.
        batch_size: Training batch size.
        num_epochs: Maximum training epochs.
        early_stopping_patience: Epochs to wait for improvement.
        temperature: InfoNCE temperature.
        top_k: Candidates to retrieve per user.
        device: Torch device string ("cpu", "mps", "cuda").
        seed: Random seed.
    """

    def __init__(
        self,
        user_dim: int = 64,
        item_dim: int = 64,
        embedding_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 2048,
        num_epochs: int = 20,
        early_stopping_patience: int = 3,
        temperature: float = 0.10,
        top_k: int = 100,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        super().__init__(name="two_tower", top_k=top_k, seed=seed)
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.temperature = temperature
        self.device_str = str(device)

        self._model: Optional[TwoTowerModel] = None
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._user_embeddings: Optional[np.ndarray] = None
        self._item_embeddings: Optional[np.ndarray] = None
        self._n_users: int = 0
        self._n_items: int = 0
        self.training_history: List[dict] = []

    @property
    def device(self) -> torch.device:
        """Resolve the computation device."""
        if self.device_str == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(
        self,
        train: pl.DataFrame,
        n_users: int,
        n_items: int,
    ) -> "TwoTowerRetriever":
        """Train Two-Tower model on user-item interaction pairs.

        Args:
            train: DataFrame with [user_idx, item_idx].
            n_users: Total number of unique users in catalog.
            n_items: Total number of unique items in catalog.

        Returns:
            Fitted retriever instance.
        """
        self._n_users = n_users
        self._n_items = n_items

        torch.manual_seed(self.seed)
        device = self.device
        log.info(f"Training TwoTower on device: {device}")

        # Build dataset
        user_arr = train["user_idx"].to_numpy().astype(np.int64)
        item_arr = train["item_idx"].to_numpy().astype(np.int64)
        full_dataset = InteractionDataset(user_arr, item_arr)

        # Compute empirical item frequency for Log-Q sampling bias correction
        item_counts = np.bincount(item_arr, minlength=n_items).astype(np.float32)
        item_probs = np.maximum(item_counts / max(len(item_arr), 1), 1e-8)
        item_log_q = torch.from_numpy(np.log(item_probs)).to(device)

        # Hold out 10% of pairs for validation loss monitoring to prevent overfitting
        n_val = max(int(len(user_arr) * 0.1), 1)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(user_arr))
        train_idx, val_idx = perm[n_val:], perm[:n_val]

        train_ds = InteractionDataset(user_arr[train_idx], item_arr[train_idx])
        val_ds = InteractionDataset(user_arr[val_idx], item_arr[val_idx])

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True if len(train_ds) >= self.batch_size else False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Build model
        self._model = TwoTowerModel(
            n_users=n_users,
            n_items=n_items,
            user_embedding_dim=self.user_dim,
            item_embedding_dim=self.item_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.embedding_dim,
            dropout=self.dropout,
            temperature=self.temperature,
        ).to(device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=1e-5
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: Optional[dict] = None

        with timer("TwoTowerRetriever.fit", samples=len(full_dataset)):
            for epoch in range(self.num_epochs):
                train_loss = self._train_epoch(
                    self._model, train_loader, optimizer, device, item_log_q=item_log_q
                )
                val_loss = self._eval_epoch(
                    self._model, val_loader, device, item_log_q=item_log_q
                )
                scheduler.step()
                lr = scheduler.get_last_lr()[0]

                self.training_history.append(
                    {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}
                )

                log.info(
                    f"Epoch {epoch + 1}/{self.num_epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr:.6f}"
                )

                # Early stopping on validation loss
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.cpu().clone() for k, v in self._model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        log.info(f"Early stopping at epoch {epoch + 1} (best_val_loss={best_val_loss:.4f})")
                        break

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(device)

        # Compute and cache all embeddings
        with timer("TwoTowerRetriever.encode_all"):
            self._compute_all_embeddings(device)

        with timer("TwoTowerRetriever.build_faiss"):
            self._build_faiss_index()

        self.is_fitted = True
        log.info(
            f"TwoTower trained: best_val_loss={best_val_loss:.4f}, "
            f"embedding_dim={self.embedding_dim}"
        )
        return self

    def _train_epoch(
        self,
        model: TwoTowerModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        item_log_q: Optional[torch.Tensor] = None,
    ) -> float:
        """Run one training epoch, returning mean loss."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for user_idx, item_idx in loader:
            if len(user_idx) <= 1:
                continue
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(user_idx, item_idx, item_log_q=item_log_q)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _eval_epoch(
        self,
        model: TwoTowerModel,
        loader: DataLoader,
        device: torch.device,
        item_log_q: Optional[torch.Tensor] = None,
    ) -> float:
        """Evaluate validation loss on held-out pairs."""
        model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for user_idx, item_idx in loader:
                if len(user_idx) <= 1:
                    continue
                user_idx = user_idx.to(device)
                item_idx = item_idx.to(device)

                loss = model(user_idx, item_idx, item_log_q=item_log_q)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _compute_all_embeddings(self, device: torch.device) -> None:
        """Encode all users and items into embedding matrices.

        Uses batch encoding to avoid OOM on large catalogs.
        """
        assert self._model is not None
        self._model.eval()

        encode_batch = 4096

        # User embeddings
        user_embs = []
        with torch.no_grad():
            for start in range(0, self._n_users, encode_batch):
                end = min(start + encode_batch, self._n_users)
                idx = torch.arange(start, end, dtype=torch.long, device=device)
                emb = self._model.encode_users(idx).cpu().numpy()
                user_embs.append(emb)
        self._user_embeddings = np.vstack(user_embs).astype(np.float32) if user_embs else np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Item embeddings
        item_embs = []
        with torch.no_grad():
            for start in range(0, self._n_items, encode_batch):
                end = min(start + encode_batch, self._n_items)
                idx = torch.arange(start, end, dtype=torch.long, device=device)
                emb = self._model.encode_items(idx).cpu().numpy()
                item_embs.append(emb)
        self._item_embeddings = np.vstack(item_embs).astype(np.float32) if item_embs else np.zeros((0, self.embedding_dim), dtype=np.float32)

        log.info(
            f"Embeddings computed: users={self._user_embeddings.shape}, "
            f"items={self._item_embeddings.shape}"
        )

    def _build_faiss_index(self) -> None:
        """Build FAISS IP (inner product = cosine after L2-norm) index."""
        assert self._item_embeddings is not None
        self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self._faiss_index.add(np.ascontiguousarray(self._item_embeddings, dtype=np.float32))
        log.info(f"FAISS index: {self._faiss_index.ntotal:,} items")

    def get_candidates(
        self,
        user_indices: List[int],
        exclude_seen: bool = True,
        seen_items: Optional[Dict[int, List[int]]] = None,
    ) -> pl.DataFrame:
        """Retrieve top-K items via FAISS MIPS on learned embeddings.

        Args:
            user_indices: List of user integer indices.
            exclude_seen: Whether to exclude seen items.
            seen_items: Dict mapping user_idx → seen item indices.

        Returns:
            DataFrame with [user_idx, item_idx, score, rank, retriever_name].
        """
        self._check_fitted()
        seen_items = seen_items or {}

        results: List[RetrievalResult] = []
        valid_uids = []

        for uid in user_indices:
            if 0 <= uid < len(self._user_embeddings):
                valid_uids.append(uid)
            else:
                results.append(
                    RetrievalResult(
                        user_idx=uid,
                        item_indices=[],
                        scores=[],
                        retriever_name=self.name,
                    )
                )

        if valid_uids:
            user_vecs = np.ascontiguousarray(self._user_embeddings[valid_uids], dtype=np.float32)
            n_retrieve = min(self.top_k * 2, self._n_items)
            scores_batch, indices_batch = self._faiss_index.search(user_vecs, n_retrieve)

            for i, user_idx in enumerate(valid_uids):
                seen = set(seen_items.get(user_idx, []))
                candidates, cand_scores = [], []

                for item_idx, score in zip(
                    indices_batch[i].tolist(), scores_batch[i].tolist()
                ):
                    if item_idx < 0 or item_idx >= self._n_items:
                        continue
                    if exclude_seen and item_idx in seen:
                        continue
                    candidates.append(item_idx)
                    cand_scores.append(float(score))
                    if len(candidates) >= self.top_k:
                        break

                results.append(
                    RetrievalResult(
                        user_idx=user_idx,
                        item_indices=candidates,
                        scores=cand_scores,
                        retriever_name=self.name,
                    )
                )

        return self._candidates_to_df(results)

    def get_user_embeddings(self) -> np.ndarray:
        """Return cached user embedding matrix."""
        self._check_fitted()
        return self._user_embeddings

    def get_item_embeddings(self) -> np.ndarray:
        """Return cached item embedding matrix."""
        self._check_fitted()
        return self._item_embeddings

    def save(self, path: Path) -> None:
        """Save model weights, embeddings, and FAISS index."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.state_dict(), path / "model.pt")
        np.save(path / "user_embeddings.npy", self._user_embeddings)
        np.save(path / "item_embeddings.npy", self._item_embeddings)
        faiss.write_index(self._faiss_index, str(path / "faiss.index"))

        config = {
            "user_dim": self.user_dim,
            "item_dim": self.item_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "temperature": self.temperature,
            "n_users": self._n_users,
            "n_items": self._n_items,
            "training_history": self.training_history,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        log.info(f"TwoTower saved to {path}")

    def load(self, path: Path) -> TwoTowerRetriever:
        """Load Two-Tower model from disk."""
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)

        self._n_users = config["n_users"]
        self._n_items = config["n_items"]
        self.training_history = config.get("training_history", [])

        self._model = TwoTowerModel(
            n_users=self._n_users,
            n_items=self._n_items,
            user_embedding_dim=self.user_dim,
            item_embedding_dim=self.item_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.embedding_dim,
            dropout=self.dropout,
            temperature=self.temperature,
        )
        self._model.load_state_dict(
            torch.load(path / "model.pt", map_location="cpu")
        )
        self._model.eval()

        self._user_embeddings = np.load(path / "user_embeddings.npy")
        self._item_embeddings = np.load(path / "item_embeddings.npy")
        self._faiss_index = faiss.read_index(str(path / "faiss.index"))
        self.is_fitted = True

        log.info(f"TwoTower loaded from {path}")
        return self
