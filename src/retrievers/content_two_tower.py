"""Content-aware Two-Tower retrieval model.

Extends the base Two-Tower with optional content modalities:
    - Text embeddings (384-dim, from sentence-transformers)
    - CLIP visual embeddings (512-dim, from CLIP ViT-B/32)

Architecture:

    User tower:
        user_id → Embedding(n_users, user_dim) → MLP → 128-dim

    Item tower (content-aware):
        item_id  → Embedding(n_items, item_dim) ──────────────────→ concat
        text_emb → Linear(384, 128) → ReLU ──────────────────────→ concat → MLP → 128-dim
        clip_emb → Linear(512, 128) → ReLU ──────────────────────→ concat

Training:
    InfoNCE loss with in-batch negatives.
    Cosine annealing learning rate schedule.

Usage:
    # Text only
    retriever = ContentTwoTowerRetriever(
        text_embeddings=text_embs,      # (n_items, 384)
        text_item_ids=text_ids,         # article_id strings
    )

    # Text + CLIP
    retriever = ContentTwoTowerRetriever(
        text_embeddings=text_embs,
        text_item_ids=text_ids,
        clip_embeddings=clip_embs,      # (n_clip_items, 512)
        clip_item_ids=clip_ids,
    )
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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

# ─── Dataset ─────────────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    """User-item interaction dataset for Two-Tower training.

    Args:
        user_indices: Array of user integer indices.
        item_indices: Array of item integer indices.
    """

    def __init__(self, user_indices: np.ndarray, item_indices: np.ndarray) -> None:
        """Initialise dataset from interaction arrays."""
        self.users = torch.from_numpy(user_indices.astype(np.int64))
        self.items = torch.from_numpy(item_indices.astype(np.int64))

    def __len__(self) -> int:
        """Return number of interactions."""
        return len(self.users)

    def __getitem__(self, idx: int):
        """Return (user_idx, item_idx) pair."""
        return self.users[idx], self.items[idx]


# ─── Model ───────────────────────────────────────────────────────────────────

class UserTower(nn.Module):
    """User embedding tower.

    Args:
        n_users: Number of users.
        user_dim: User embedding lookup dimension.
        hidden_dims: MLP hidden layer sizes.
        output_dim: Final embedding dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_users: int,
        user_dim: int = 128,
        hidden_dims: list[int] = [256, 128],
        output_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialise user tower."""
        super().__init__()
        self.embedding = nn.Embedding(n_users, user_dim)

        layers: list[nn.Module] = []
        in_dim = user_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass returning L2-normalised user vectors.

        Args:
            user_idx: (batch,) integer tensor of user indices.

        Returns:
            (batch, output_dim) normalised embedding tensor.
        """
        x = self.embedding(user_idx)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class ContentItemTower(nn.Module):
    """Content-aware item tower combining ID + text + visual embeddings.

    Args:
        n_items: Number of items.
        item_dim: Item ID embedding dimension.
        text_embeddings: Optional (n_items, text_dim) text embedding matrix.
        clip_embeddings: Optional (n_clip, clip_dim) CLIP embedding matrix.
        clip_item_mask: Boolean mask of which items have CLIP embeddings.
        hidden_dims: MLP hidden dimensions.
        output_dim: Final output dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_items: int,
        item_dim: int = 128,
        text_embeddings: Optional[torch.Tensor] = None,
        clip_embeddings: Optional[torch.Tensor] = None,
        clip_item_mask: Optional[torch.Tensor] = None,
        content_proj_dim: int = 128,
        hidden_dims: list[int] = [256, 128],
        output_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialise content-aware item tower."""
        super().__init__()

        self.has_text = text_embeddings is not None
        self.has_clip = clip_embeddings is not None

        self.embedding = nn.Embedding(n_items, item_dim)

        # Text projection
        if self.has_text:
            assert text_embeddings is not None
            text_dim = text_embeddings.shape[1]
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, content_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # Register as buffer (not trained, but moves with model.to(device))
            self.register_buffer("text_emb", text_embeddings.float())

        # CLIP projection
        if self.has_clip:
            assert clip_embeddings is not None
            assert clip_item_mask is not None
            clip_dim = clip_embeddings.shape[1]
            self.clip_proj = nn.Sequential(
                nn.Linear(clip_dim, content_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.register_buffer("clip_emb", clip_embeddings.float())
            self.register_buffer("clip_mask", clip_item_mask.float())

        # Compute total input dim to MLP
        fusion_dim = item_dim
        if self.has_text:
            fusion_dim += content_proj_dim
        if self.has_clip:
            fusion_dim += content_proj_dim

        # Fusion MLP
        layers: list[nn.Module] = []
        in_dim = fusion_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, item_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass combining ID + content embeddings.

        Args:
            item_idx: (batch,) integer tensor of item indices.

        Returns:
            (batch, output_dim) normalised embedding tensor.
        """
        parts = [self.embedding(item_idx)]

        if self.has_text:
            text_feats = self.text_emb[item_idx]  # (batch, text_dim)
            parts.append(self.text_proj(text_feats))

        if self.has_clip:
            clip_feats = self.clip_emb[item_idx]   # (batch, clip_dim)
            mask = self.clip_mask[item_idx].unsqueeze(-1)  # (batch, 1)
            projected = self.clip_proj(clip_feats) * mask  # zero if no image
            parts.append(projected)

        x = torch.cat(parts, dim=-1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class ContentTwoTowerModel(nn.Module):
    """Full content-aware Two-Tower model.

    Args:
        n_users: Number of users.
        n_items: Number of items.
        user_dim: User embedding dimension.
        item_dim: Item ID embedding dimension.
        text_embeddings: Optional text embedding matrix.
        clip_embeddings: Optional CLIP embedding matrix.
        clip_item_mask: Boolean item mask for CLIP coverage.
        hidden_dims: MLP hidden dimensions for both towers.
        output_dim: Final shared embedding dimension.
        dropout: Dropout probability.
        temperature: InfoNCE temperature.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_dim: int = 128,
        item_dim: int = 128,
        text_embeddings: Optional[torch.Tensor] = None,
        clip_embeddings: Optional[torch.Tensor] = None,
        clip_item_mask: Optional[torch.Tensor] = None,
        hidden_dims: list[int] = [256, 128],
        output_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        """Initialise full Two-Tower model."""
        super().__init__()
        self.temperature = temperature

        self.user_tower = UserTower(
            n_users=n_users,
            user_dim=user_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.item_tower = ContentItemTower(
            n_items=n_items,
            item_dim=item_dim,
            text_embeddings=text_embeddings,
            clip_embeddings=clip_embeddings,
            clip_item_mask=clip_item_mask,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss for a batch of (user, item) pairs.

        Args:
            user_idx: (batch,) user indices.
            item_idx: (batch,) item indices.

        Returns:
            Scalar InfoNCE loss.
        """
        user_emb = self.user_tower(user_idx)    # (B, D)
        item_emb = self.item_tower(item_idx)    # (B, D)

        # Similarity matrix: (B, B)
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature

        # Diagonal = positive pairs
        labels = torch.arange(len(user_idx), device=user_idx.device)
        loss = F.cross_entropy(logits, labels)
        return loss


# ─── Retriever ───────────────────────────────────────────────────────────────

class ContentTwoTowerRetriever(BaseRetriever):
    """Content-aware Two-Tower retriever.

    Wraps ContentTwoTowerModel with FAISS retrieval,
    training loop, and checkpoint I/O.

    Args:
        text_embeddings: Optional (n_items, 384) text embedding matrix.
        text_item_ids: Article IDs corresponding to text_embeddings rows.
        clip_embeddings: Optional (n_clip, 512) CLIP embedding matrix.
        clip_item_ids: Article IDs corresponding to clip_embeddings rows.
        embedding_dim: Output embedding dimension.
        hidden_dims: MLP hidden layers.
        user_dim: User ID embedding size.
        item_dim: Item ID embedding size.
        num_epochs: Training epochs.
        batch_size: Training batch size.
        learning_rate: Peak learning rate.
        temperature: InfoNCE temperature.
        top_k: Number of candidates to retrieve per user.
        device: Torch device.
        seed: Random seed.
    """

    def __init__(
        self,
        text_embeddings: Optional[np.ndarray] = None,
        text_item_ids: Optional[np.ndarray] = None,
        clip_embeddings: Optional[np.ndarray] = None,
        clip_item_ids: Optional[np.ndarray] = None,
        embedding_dim: int = 128,
        hidden_dims: list[int] = [256, 128],
        user_dim: int = 128,
        item_dim: int = 128,
        num_epochs: int = 20,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        temperature: float = 0.07,
        top_k: int = 100,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        """Initialise content-aware Two-Tower retriever."""
        super().__init__(name="content_two_tower", top_k=top_k)

        self.text_embeddings = text_embeddings
        self.text_item_ids = text_item_ids
        self.clip_embeddings = clip_embeddings
        self.clip_item_ids = clip_item_ids

        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.device = device
        self.seed = seed

        self._model: Optional[ContentTwoTowerModel] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._user_embeddings: Optional[np.ndarray] = None
        self._item_embeddings: Optional[np.ndarray] = None
        self._n_users = 0
        self._n_items = 0
        self._training_history: list[dict] = []

    def _build_content_tensors(
        self,
        item2idx: Dict[str, int],
        n_items: int,
    ):
        """Build aligned content embedding tensors for the item tower.

        Args:
            item2idx: Mapping from article_id string to integer index.
            n_items: Total number of items in the catalog.

        Returns:
            Tuple of (text_tensor, clip_tensor, clip_mask_tensor).
            Any of these may be None if the modality is unavailable.
        """
        text_tensor = None
        clip_tensor = None
        clip_mask = None

        # Text embeddings — aligned to item index
        if self.text_embeddings is not None and self.text_item_ids is not None:
            text_dim = self.text_embeddings.shape[1]
            aligned_text = np.zeros((n_items, text_dim), dtype=np.float32)

            text_id_to_row = {
                str(aid).zfill(10): i for i, aid in enumerate(self.text_item_ids)
            }
            hit = 0
            for article_id, item_idx in item2idx.items():
                row = text_id_to_row.get(str(article_id).zfill(10))
                if row is not None:
                    aligned_text[item_idx] = self.text_embeddings[row]
                    hit += 1

            log.info(f"Text embeddings aligned: {hit}/{n_items} items ({100*hit/n_items:.1f}%)")
            text_tensor = torch.from_numpy(aligned_text)

        # CLIP embeddings — aligned to item index, with coverage mask
        if self.clip_embeddings is not None and self.clip_item_ids is not None:
            clip_dim = self.clip_embeddings.shape[1]
            aligned_clip = np.zeros((n_items, clip_dim), dtype=np.float32)
            clip_mask_arr = np.zeros(n_items, dtype=np.float32)

            clip_id_to_row = {
                str(aid): i for i, aid in enumerate(self.clip_item_ids)
            }
            hit = 0
            for article_id, item_idx in item2idx.items():
                row = clip_id_to_row.get(str(article_id))
                if row is not None:
                    aligned_clip[item_idx] = self.clip_embeddings[row]
                    clip_mask_arr[item_idx] = 1.0
                    hit += 1

            log.info(f"CLIP embeddings aligned: {hit}/{n_items} items ({100*hit/n_items:.1f}%)")
            clip_tensor = torch.from_numpy(aligned_clip)
            clip_mask = torch.from_numpy(clip_mask_arr)

        return text_tensor, clip_tensor, clip_mask

    @timer("ContentTwoTowerRetriever.fit")
    def fit(
        self,
        interactions: pl.DataFrame,
        n_users: int,
        n_items: int,
        item2idx: Optional[Dict[str, int]] = None,
    ) -> "ContentTwoTowerRetriever":
        """Train the content-aware Two-Tower model.

        Args:
            interactions: DataFrame with [user_idx, item_idx] columns.
            n_users: Total number of users.
            n_items: Total number of items.
            item2idx: Optional mapping from article_id to item index.

        Returns:
            Self, for chaining.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._n_users = n_users
        self._n_items = n_items

        # Build content tensors
        text_tensor, clip_tensor, clip_mask = None, None, None
        if item2idx is not None:
            text_tensor, clip_tensor, clip_mask = self._build_content_tensors(
                item2idx, n_items
            )

        modalities = ["ID"]
        if text_tensor is not None:
            modalities.append("text(384-dim)")
        if clip_tensor is not None:
            modalities.append("CLIP(512-dim)")
        log.info(f"Training ContentTwoTower on device: {self.device}")
        log.info(f"Active modalities: {' + '.join(modalities)}")

        # Build model
        self._model = ContentTwoTowerModel(
            n_users=n_users,
            n_items=n_items,
            user_dim=self.user_dim,
            item_dim=self.item_dim,
            text_embeddings=text_tensor,
            clip_embeddings=clip_tensor,
            clip_item_mask=clip_mask,
            hidden_dims=self.hidden_dims,
            output_dim=self.embedding_dim,
        ).to(self.device)

        # Dataset and dataloader
        user_arr = interactions["user_idx"].to_numpy().astype(np.int64)
        item_arr = interactions["item_idx"].to_numpy().astype(np.int64)
        dataset = InteractionDataset(user_arr, item_arr)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Required on macOS
            drop_last=True,
        )

        # Optimizer + cosine annealing
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs
        )

        best_loss = float("inf")
        self._training_history = []

        for epoch in range(1, self.num_epochs + 1):
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0

            for user_batch, item_batch in dataloader:
                user_batch = user_batch.to(self.device)
                item_batch = item_batch.to(self.device)

                optimizer.zero_grad()
                loss = self._model(user_batch, item_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            lr = scheduler.get_last_lr()[0]
            scheduler.step()

            if avg_loss < best_loss:
                best_loss = avg_loss

            self._training_history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "lr": lr,
            })
            log.info(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"loss={avg_loss:.4f} | lr={lr:.6f}"
            )

        log.info(f"Training complete. Best loss: {best_loss:.4f}")

        # Compute and index all embeddings
        self._compute_all_embeddings()
        self._build_faiss_index()
        self._is_fitted = True
        return self

    def _compute_all_embeddings(self) -> None:
        """Compute user and item embeddings for the full catalog using batched evaluation."""
        assert self._model is not None
        self._model.eval()

        encode_batch = 4096

        # User embeddings
        user_embs = []
        with torch.no_grad():
            for start in range(0, self._n_users, encode_batch):
                end = min(start + encode_batch, self._n_users)
                idx = torch.arange(start, end, dtype=torch.long, device=self.device)
                emb = self._model.user_tower(idx).cpu().numpy()
                user_embs.append(emb)
        self._user_embeddings = np.vstack(user_embs).astype(np.float32) if user_embs else np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Item embeddings
        item_embs = []
        with torch.no_grad():
            for start in range(0, self._n_items, encode_batch):
                end = min(start + encode_batch, self._n_items)
                idx = torch.arange(start, end, dtype=torch.long, device=self.device)
                emb = self._model.item_tower(idx).cpu().numpy()
                item_embs.append(emb)
        self._item_embeddings = np.vstack(item_embs).astype(np.float32) if item_embs else np.zeros((0, self.embedding_dim), dtype=np.float32)

        log.info(
            f"Embeddings computed: users={self._user_embeddings.shape}, "
            f"items={self._item_embeddings.shape}"
        )

    def _build_faiss_index(self) -> None:
        """Build FAISS FlatIP index over item embeddings."""
        assert self._item_embeddings is not None
        dim = self._item_embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(np.ascontiguousarray(self._item_embeddings, dtype=np.float32))
        log.info(f"FAISS index: {self._item_embeddings.shape[0]} items")

    def get_candidates(
        self,
        user_indices: list[int],
        exclude_seen: bool = True,
        seen_items: Optional[Dict[int, list[int]]] = None,
    ) -> pl.DataFrame:
        """Retrieve top-K candidates for each user via FAISS.

        Args:
            user_indices: List of user integer indices.
            exclude_seen: Whether to exclude seen items.
            seen_items: Dict mapping user_idx to seen item indices.

        Returns:
            Polars DataFrame with [user_idx, item_idx, score, rank].
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_candidates()")

        assert self._faiss_index is not None
        assert self._user_embeddings is not None

        seen = seen_items or {}
        rows = []
        valid_uids = []

        for uid in user_indices:
            if 0 <= uid < len(self._user_embeddings):
                valid_uids.append(uid)

        if valid_uids:
            user_embs = np.ascontiguousarray(self._user_embeddings[valid_uids], dtype=np.float32)
            fetch_k = min(self.top_k * 2, self._n_items)
            scores_matrix, items_matrix = self._faiss_index.search(user_embs, fetch_k)

            for i, user_idx in enumerate(valid_uids):
                user_seen = set(seen.get(user_idx, []))
                rank = 1
                for item_idx, score in zip(items_matrix[i], scores_matrix[i]):
                    if item_idx < 0 or item_idx >= self._n_items:
                        continue
                    if exclude_seen and item_idx in user_seen:
                        continue
                    rows.append({
                        "user_idx": user_idx,
                        "item_idx": int(item_idx),
                        "score": float(score),
                        "rank": rank,
                        "retriever_name": self.name,
                    })
                    rank += 1
                    if rank > self.top_k:
                        break

        return pl.DataFrame(rows) if rows else pl.DataFrame(
            schema={"user_idx": pl.Int32, "item_idx": pl.Int32,
                    "score": pl.Float32, "rank": pl.Int32,
                    "retriever_name": pl.Utf8}
        )

    def get_user_embeddings(self) -> np.ndarray:
        """Return user embedding matrix."""
        assert self._user_embeddings is not None
        return self._user_embeddings

    def get_item_embeddings(self) -> np.ndarray:
        """Return item embedding matrix."""
        assert self._item_embeddings is not None
        return self._item_embeddings

    def _build_seen_items(self, train: pl.DataFrame) -> Dict[int, list[int]]:
        """Build seen items dictionary from training interactions."""
        seen: Dict[int, list[int]] = {}
        for row in train.select(["user_idx", "item_idx"]).iter_rows():
            uid, iid = row
            if uid not in seen:
                seen[uid] = []
            seen[uid].append(iid)
        return seen

    def save(self, path: Path) -> None:
        """Save model and embeddings to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self._model is not None

        torch.save(self._model.state_dict(), path / "model.pt")
        np.save(path / "user_embeddings.npy", self._user_embeddings)
        np.save(path / "item_embeddings.npy", self._item_embeddings)
        faiss.write_index(self._faiss_index, str(path / "faiss.index"))

        config = {
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "user_dim": self.user_dim,
            "item_dim": self.item_dim,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "device": self.device,
            "n_users": self._n_users,
            "n_items": self._n_items,
            "training_history": self._training_history,
            "has_text": self.text_embeddings is not None,
            "has_clip": self.clip_embeddings is not None,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        log.info(f"ContentTwoTower saved to {path}")

    def load(self, path: Path) -> ContentTwoTowerRetriever:
        """Load model from disk."""
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)

        self._n_users = config["n_users"]
        self._n_items = config["n_items"]
        self._user_embeddings = np.load(path / "user_embeddings.npy")
        self._item_embeddings = np.load(path / "item_embeddings.npy")
        self._faiss_index = faiss.read_index(str(path / "faiss.index"))
        self._training_history = config.get("training_history", [])

        # Rebuild model structure and restore weights if model.pt exists
        model_pt = path / "model.pt"
        if model_pt.exists():
            text_tensor = torch.from_numpy(self.text_embeddings) if self.text_embeddings is not None else None
            clip_tensor = torch.from_numpy(self.clip_embeddings) if self.clip_embeddings is not None else None
            clip_mask = torch.from_numpy(self.clip_item_mask) if self.clip_item_mask is not None else None
            self._model = ContentTwoTowerModel(
                n_users=self._n_users,
                n_items=self._n_items,
                user_dim=self.user_dim,
                item_dim=self.item_dim,
                text_embeddings=text_tensor,
                clip_embeddings=clip_tensor,
                clip_item_mask=clip_mask,
                hidden_dims=self.hidden_dims,
                output_dim=self.embedding_dim,
                temperature=self.temperature,
            )
            self._model.load_state_dict(torch.load(model_pt, map_location="cpu"))
            self._model.eval()

        self._is_fitted = True
        log.info(f"ContentTwoTower loaded from {path}")
        return self
