import os

# Prevent OpenMP / PyTorch / BLAS thread contention and crashes on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    from tqdm import tqdm
    tqdm.monitor_interval = 0
except ImportError:
    pass
