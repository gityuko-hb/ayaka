"""The reference backend: every op in plain PyTorch, on CPU and CUDA.

It is the numerical reference the other backends are compared with, and what
hosted CI runs. It is written to be read: there is no fast path here, and it is
not the backend to measure speed with.

Each family module implements its contract with module-level functions, so the
module itself is the implementation ``load_backend`` hands to the layers. The
modules import torch at the top; import them through ``load_backend``.
"""
