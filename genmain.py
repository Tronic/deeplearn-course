import facedata
import numpy as np
import torch
import torch.nn as nn
import time
import latent
import visualization

#%% Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
faces = facedata.Torch(device=device)

base_size = 10
image_size = base_size << 5  # 160px max size
discriminator_channels = 64
generator_channels = 64

#%% Network definition


class DownConvLayer(nn.Module):
    """A layer used in discriminator"""
    def __init__(self, channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.LeakyReLU(0.25, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.MaxPool2d(2), nn.LeakyReLU(0.25, inplace=True),
        )

    def forward(self, x):
        return self.seq(x)

class Discriminator(nn.Module):
    """Discriminates face images into real and fake; return values more negative indicate fake and more positive indicate real."""
    def __init__(self, base_size=base_size, channels=discriminator_channels):
        super().__init__()
        self.fromRGB = nn.ConvTranspose2d(3, channels, kernel_size=1)
        self.conv = nn.ModuleList([DownConvLayer(channels) for i in range(5)])
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.Tanh(),
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.seq(x)


class LatentIn(nn.Module):
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
        self.seq = nn.Sequential(nn.Sigmoid(), nn.ConvTranspose2d(ch, 3, kernel_size=1), nn.Tanh())

    def forward(self, x):
        return self.seq(x)

class Generator(nn.Module):
    """Convolutional generator adapted from DCGAN by Radford et al."""
    def __init__(self, base_size=base_size, channels=generator_channels):
        super().__init__()
        n_layers = 5
        self.channels = channels
        self.base_size = base_size
        self.latimg = nn.Linear(latent.dimension, channels * base_size**2, bias=False)
        self.lat = LatentIn(image_size=base_size << n_layers, channels=channels)
        self.upc = nn.ModuleList([UpConvLayer(2 * channels, channels) for i in range(n_layers)])
        self.toRGB = ToRGB(channels)  # Map channels to colors

    def forward(self, latent, image_size=image_size, alpha=1, train=False):
        x = self.latimg(latent).view(-1, self.channels, self.base_size, self.base_size)
        lat_full = self.lat(latent)
        for i, upconv in enumerate(self.upc):
            if x.size(2) >= image_size: break
            lat = nn.functional.interpolate(lat_full, x.size(2))
            x_prev, x = x, upconv(torch.cat((x, lat), dim=1))
            # Minimize correlation between samples
            if train:
                x0 = x.detach()
                x1 = torch.cat((x0[1:], x0[0:1]), dim=0)
                xd = x0 - x1
                x.backward(xd.sign() / (abs(xd) + 0.1)**2, retain_graph=True)
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


#%% Init GAN
discriminator = Discriminator()
generator = Generator()

generator.to(device)
discriminator.to(device)

#%% Load previous state
try:
    import os, glob
    filename = sorted(glob.glob("facegen*.pth"), key=os.path.getmtime)[-1]
    checkpoint = torch.load(filename, map_location=lambda storage, location: storage)
    discriminator.load_state_dict(checkpoint["discriminator"])
    generator.load_state_dict(checkpoint["generator"])
    print("Networks loaded:", filename)
except:
    pass

#%% Training
minibatch_size = 16
rounds = range(64000 // minibatch_size if device.type == "cuda" else 10)
epochs = range(6)

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(.5, .999))
g_optimizer = torch.optim.Adam([
    {"params": generator.latimg.parameters(), "lr": 0.0005},
    {"params": generator.lat.parameters(), "lr": 0.0005},
    {"params": generator.upc.parameters(), "lr": 0.00003},
    {"params": generator.toRGB.parameters(), "lr": 0.001},
], betas=(.8, .99))

def training():
    print(f"Training with {len(rounds)} rounds per epoch:")
    ones = torch.ones((minibatch_size, 1), device=device)
    zeros = torch.zeros((minibatch_size, 1), device=device)
    criterion = nn.BCEWithLogitsLoss()
    isize = base_size
    for e in epochs:
        if isize < image_size: isize *= 2
        images = faces.batch_iter(minibatch_size, image_size=isize)
        d_rounds = g_rounds = 0
        rtimer = time.perf_counter()
        for r in rounds:
            alpha = min(1.0, 4 * r / len(rounds))  # Alpha blending after switching to bigger resolution
            # Make a set of fakes
            z = latent.random(minibatch_size, device=device)
            g_optimizer.zero_grad()
            fake = generator(z, image_size=isize, alpha=alpha, train=True)
            # Train the generator
            loss = criterion(discriminator(fake, alpha), ones)
            fake = fake.detach()  # Drop gradients (we don't want more generator updates)
            loss.backward()
            g_optimizer.step()
            g_rounds += 1
            # Train the discriminator one time or until it is good enough
            for real in images:  # Infinite loop of random sample images
                assert real.shape == fake.shape
                # Discriminate real and fake images
                d_optimizer.zero_grad()
                output_fake = discriminator(fake, alpha)
                output_real = discriminator(real, alpha)
                # Train the discriminator
                criterion(output_real, ones).backward()
                criterion(output_fake, zeros).backward()
                d_optimizer.step()
                d_rounds += 1
                # Check levels
                level_real, level_fake = torch.sigmoid(torch.stack([
                    output_real.detach(),
                    output_fake.detach()
                ])).view(2, minibatch_size).mean(dim=1).cpu().numpy()
                level_diff = level_real - level_fake
                glr = g_optimizer.param_groups[0]['lr']
                dlr = d_optimizer.param_groups[0]['lr']
                stats = f"{g_rounds:04d}:{d_rounds:04d}  lr={glr*1e6:03.0f}:{dlr*1e6:03.0f}µ {level_real:4.0%} vs{level_fake:4.0%}  {(time.perf_counter() - rtimer) / (r + .1) * len(rounds):3.0f} s/epoch"
                bar = f"[{'*' * (25 * r // rounds[-1]):25s}]"
                alp = 4 * " ░▒▓█"[int(alpha * 4)]
                print(f"\r  {bar} {stats} {alp}", end="\N{ESC}[K")
                if level_diff > 0.2: break  # Good enough
                #for param_group in d_optimizer.param_groups: param_group['lr'] *= 1.1
            if level_diff > 0.99:
                for param_group in d_optimizer.param_groups: param_group['lr'] *= 0.99

            visualize(f"{isize:3}px {bar} alpha {alp} »  G:D {stats}", image_size=isize, alpha=alpha)
        print(f"\r  Epoch {e:2}/{len(epochs)} {isize:3}px done   » {stats}", end="\N{ESC}[K\n")
        torch.save({
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        }, f"facegen{e:03}.pth")
        # After each epoch, reduce generator learning rates
        for param_group in g_optimizer.param_groups: param_group['lr'] *= 0.5

with visualization.Video(generator, device=device) as visualize:
    training()
