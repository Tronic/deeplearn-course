import torch

dimension = 100

def random(*size, device=None):
    """Create random latent variables for size samples."""
    global dimension
    z = torch.randn((*size, dimension), device=device)
    lengths = (z * z).sum(dim=-1, keepdim=True)**.5
    return z / lengths * dimension**.5  # Form a hypersphere of radius sqrt(zdim)
