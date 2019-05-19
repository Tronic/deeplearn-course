import torch

dimension = 100

def random(*size, device=None):
    """Create random latent variables for size samples."""
    global dimension
    z = torch.randn((*size, dimension), device=device)
    lengths = (z * z).sum(dim=-1, keepdim=True)**.5
    return z / lengths * dimension**.5  # Form a hypersphere of radius sqrt(zdim)

def slerp(l1: torch.Tensor, l2: torch.Tensor, t: float):
    """Perform spherical interpolation between two latent tensors."""
    assert 0 <= t <= 1
    assert l1.shape == l2.shape
    assert l1.size(-1) == dimension
    o = torch.acos((t1 * t2).sum(dim=-1) / (2 * t1.norm(dim=-1) * t2.norm(dim=-1)))
    return (torch.sin((1 - t) * o) * t1 + torch.sin(t * o) * t2) / torch.sin(o)
