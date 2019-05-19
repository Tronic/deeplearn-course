import torch
import torch.nn as nn
import latent

# Network settings
discriminator_channels = 32
generator_channels = 128
n_layers = 3
max_size = 160  # px
base_size = max_size >> n_layers


class DownConvLayer(nn.Module):
    """A layer used in generator; convolutes and avgpools input into half size."""
    def __init__(self, channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.LeakyReLU(0.25, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.AvgPool2d(2), nn.LeakyReLU(0.25, inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class Discriminator(nn.Module):
    """Discriminates face images into real and fake; return values more negative indicate fake and more positive indicate real."""
    def __init__(self, base_size=base_size, channels=discriminator_channels):
        super().__init__()
        self.fromRGB = nn.ConvTranspose2d(3, channels, kernel_size=1)
        self.conv = nn.ModuleList([DownConvLayer(channels) for i in range(n_layers)])
        self.linear = nn.Sequential(
            nn.Linear(channels * base_size * base_size, 256), nn.LeakyReLU(0.25, inplace=True),
            nn.Linear(256, 128), nn.LeakyReLU(0.25, inplace=True),
            nn.Linear(128, 64), nn.LeakyReLU(0.25, inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, faces, alpha=1):
        # The output is a sum of normal and horizontally mirrored faces' scores
        return sum(self.discriminate(x, alpha) for x in [faces, faces.flip(dims=(3,))])

    def discriminate(self, x, alpha=1):
        N, ch, h, w = x.shape
        n_layers = (h // base_size).bit_length() - 1
        # RGB conversion, convolute down to base size, then linear layers
        x = self.fromRGB(x)
        for dc in reversed(self.conv[:n_layers]):
            if alpha < 1:
                xs = nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
                x = torch.lerp(xs, dc(x), alpha)
                alpha = 1  # No blending for other layers
            else:
                x = dc(x)
        assert x.size(2) == base_size, f"Image size {h} -> {x.size(2)}, should be -> {base_size}."
        return self.linear(x.view(N, -1))


class UpConvLayer(nn.Module):
    """A layer used in generator; upscales input to double size and convolutes."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.Tanh(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.Tanh(),
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.seq(x)


class Inject(nn.Module):
    def __init__(self, image_size, channels=latent.dimension):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.seq = nn.Sequential(nn.Linear(latent.dimension, channels - 2, bias=False), nn.LeakyReLU(0.125, inplace=True))
        ls = torch.linspace(-10.0, 10.0, image_size)
        self.register_buffer("coords", torch.cat((
            ls.view(1, 1, self.image_size, 1).expand(-1, -1, -1, self.image_size),
            ls.view(1, 1, 1, self.image_size).expand(-1, -1, self.image_size, -1),
        ), dim=1))

    def forward(self, x):
        x = self.seq(x)
        x = x.view(-1, self.channels - 2, 1, 1).expand(-1, -1, self.image_size, self.image_size)
        return torch.cat((x, self.coords.expand(x.size(0), -1, -1, -1)), dim=1)

class ToRGB(nn.Module):
    """Convert channels into colors."""
    def __init__(self, ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, kernel_size=1), nn.Tanh(),
            nn.ConvTranspose2d(ch, 3, kernel_size=1), nn.Tanh(),
        )

    def forward(self, x):
        return self.seq(x)

class Generator(nn.Module):
    """Convolutional generator adapted from DCGAN by Radford et al."""
    def __init__(self, base_size=base_size, channels=generator_channels):
        super().__init__()
        self.channels = channels
        self.base_size = base_size
        self.latimg = nn.Linear(latent.dimension, channels * base_size**2, bias=False)
        self.inject = Inject(image_size=base_size << n_layers, channels=channels)
        self.upc = nn.ModuleList([UpConvLayer(2 * channels, channels) for i in range(n_layers)])
        self.toRGB = ToRGB(channels)  # Map channels to colors

    def forward(self, latent, image_size=max_size, alpha=1, diversify=0):
        x = 0 * self.latimg(latent).view(-1, self.channels, self.base_size, self.base_size)
        lat_full = self.inject(latent)
        size = base_size
        for i, upconv in enumerate(self.upc):
            size *= 2
            if size >= image_size: break
            lat = nn.functional.interpolate(lat_full, size)
            x_prev, x = x, upconv(torch.cat((x, lat), dim=1))
        # Minimize correlation between samples
        if diversify:
            with torch.no_grad():
                g = torch.cat((x[1:], x[0:1]), dim=0) - x
                g = g.sign() * .05 * (1.05 / (g**2 + .05) - 1)
                x.backward(alpha * diversify / g.numel() * g, retain_graph=True)
                del g
        # Alpha blending between the last two layers
        if alpha < 1 and "x_prev" in locals():
            x_prev = nn.functional.interpolate(x_prev, scale_factor=2, mode="bilinear", align_corners=False)
            x = torch.lerp(x_prev, x, alpha)
        return self.toRGB(x)

# Verify network output shapes
g_output = Generator()(latent.random(10), image_size=base_size)
assert g_output.shape == torch.Size((10, 3, base_size, base_size)), f"Generator output {g_output.shape}"
d_output = Discriminator()(g_output)
assert d_output.shape == torch.Size((10, 1)), f"Discriminator output {d_output.shape}"
del g_output, d_output
