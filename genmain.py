import facedata
import numpy as np
import torch
import torch.nn as nn
import time
import latent
import visualization

#%% Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Uploading tensors to", device)
images = facedata.torch_tensor(stride=2, device=device)
assert images.shape == torch.Size((facedata.N, 3, 100, 100))

#%% Network definition

class Reshape(nn.Module):
    """Reshape image dimensions."""
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(x.size(0), *self.dims)


class Discriminator(nn.Module):
    """Output probability of face being real (1) vs. generated (0)."""
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1, inplace=True),
            Reshape(-1),
            nn.Linear(64 * 12 * 12, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, faces):
        return self.seq(faces)

class UpConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels, affine=False, momentum=0.0001),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2, stride=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=False, momentum=0.0001),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x * (x[0].numel() / (x.detach().sum() + 1.0))  # Normalize
        return self.seq(x)


class LatentIn(nn.Module):
    def __init__(self, image_size, channels=latent.dimension):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.seq = nn.Sequential(
            nn.Linear(latent.dimension, latent.dimension, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent.dimension, latent.dimension, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent.dimension, channels, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.seq(x).view(-1, self.channels, 1, 1).expand(-1, -1, self.image_size, self.image_size)


class Generator(nn.Module):
    """Convolutional generator adapted from DCGAN by Radford et al."""
    def __init__(self):
        super().__init__()
        CH, SIZE = 512, 4  # Initial channel count and image resolution
        self.lat = nn.ModuleList([
            LatentIn(image_size=SIZE, channels=CH),
            LatentIn(7, 8),
            LatentIn(13, 8),
            LatentIn(25, 8),
            LatentIn(49, 8),
            LatentIn(97, 8),
        ])
        self.upc = nn.ModuleList([
            UpConvLayer(CH + 3, CH // 2),  # out 7x7
            UpConvLayer(CH // 2 + 8, CH // 4),  # out 13x13
            UpConvLayer(CH // 4 + 8, CH // 4),  # out 25x25
            UpConvLayer(CH // 4 + 8, CH // 4),  # out 49x49
            UpConvLayer(CH // 4 + 8, CH // 4),  # out 97x97
        ])
        # Map channels to colors
        self.toRGB = nn.Sequential(
            nn.ConvTranspose2d(in_channels=CH // 4 + 8, out_channels=3, kernel_size=1, bias=True),
            nn.Tanh()
        )

    def forward(self, latent):
        device = next(self.parameters()).device
        N = latent.size(0)
        x = torch.linspace(-1, 1, 4, device=device)
        xc = x.view(1, 1, -1, 1).expand(N, 1, 4, 4)
        yc = x.view(1, 1, 1, -1).expand(N, 1, 4, 4)
        rnd = torch.randn((N, 1, 4, 4), device=device)
        x = 100.0 * torch.cat((xc, yc, rnd), dim=1)
        for upc, lat in zip(self.upc, self.lat):
            x = upc(torch.cat((x, lat(latent)), dim=1))
        x = self.toRGB(torch.cat((x, self.lat[-1](latent)), dim=1))
        assert x.size(-1) == 97, f"got {x.size(-1)} pixels wide"
        return nn.functional.interpolate(x, scale_factor=100/x.size(-1) + 1e-5, mode="bilinear", align_corners=False)

# Verify network output shapes
g_output = Generator()(latent.random(10))
assert g_output.shape == torch.Size((10, 3, 100, 100)), f"Generator output {g_output.shape}"
d_output = Discriminator()(g_output)
assert d_output.shape == torch.Size((10, 1)), f"Discriminator output {d_output.shape}"
del g_output, d_output


#%% Init GAN
visualize = visualization.PNG(device=device)

discriminator = Discriminator()
generator = Generator()

generator.to(device)
discriminator.to(device)

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(.5, .999))
g_optimizer = torch.optim.Adam([
    {"params": generator.lat.parameters(), "lr": 0.000005},
    {"params": generator.upc.parameters(), "lr": 0.00001},
    {"params": generator.toRGB.parameters(), "lr": 0.0001},
], betas=(.5, .999))

criterion = nn.BCEWithLogitsLoss()

#%% Load previous state
try:
    from glob import glob
    filename = glob("facegen*.pth")[-1]
    checkpoint = torch.load(filename, map_location=lambda storage, location: storage)
    discriminator.load_state_dict(checkpoint["discriminator"])
    generator.load_state_dict(checkpoint["generator"])
    print("Networks loaded:", filename)
    d_optimizer.load_state_dict(checkpoint["d_optimizer"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer"])
except:
    pass


#%% Training
minibatch_size = 64
rounds = range(facedata.N // minibatch_size if device.type == "cuda" else 10)
epochs = range(100)
ones = torch.ones((minibatch_size, 1), device=device)
zeros = torch.zeros((minibatch_size, 1), device=device)
level_real = level_fake = 0.5

print(f"Training with {len(rounds)} rounds per epoch:")
for e in epochs:
    d_rounds = g_rounds = 0
    rtimer = time.perf_counter()
    for r in rounds:
        print(f"  [{'*' * (25 * r // rounds[-1]):25s}] {r+1:4d}/{len(rounds)} {(time.perf_counter() - rtimer) / (r + .1) * len(rounds):3.0f} s/epoch {level_real:.0%} vs {level_fake:.0%}", end="\r")
        # Make a set of fakes
        z = latent.random(minibatch_size, device=device)
        g_optimizer.zero_grad()
        generator.train()  # Allow updates of batch normalization
        fake = generator(z)
        generator.eval()  # Disable further batchnorm updates
        # Train the discriminator until it is good enough
        while True:
            # Choose a random slice of images
            batch = np.random.randint(facedata.N - minibatch_size)
            batch = slice(batch, batch + minibatch_size)
            real = images[batch].to(torch.float32)
            real /= 128.0
            real -= 1.0
            # Discriminate real and fake images
            d_optimizer.zero_grad()
            output_fake = discriminator(fake.detach())
            output_real = discriminator(real)
            # Check levels
            levels = np.stack((
                output_real.detach().cpu().numpy().reshape(minibatch_size),
                output_fake.detach().cpu().numpy().reshape(minibatch_size)
            ))
            levels = 1.0 / (1.0 + np.exp(-levels))  # Sigmoid for probability
            level_real, level_fake = levels.mean(axis=1)
            level_diff = level_real - level_fake
            if level_diff > 0.3: break  # Good enough discrimination!
            # Train the discriminator
            criterion(output_real, ones).backward()
            criterion(output_fake, zeros).backward()
            d_optimizer.step()
            d_rounds += 1
        # Train the generator
        criterion(discriminator(fake), ones).backward()
        g_optimizer.step()
        g_rounds += 1
        visualize(generator=generator, discriminator=discriminator)
    print(f"  Epoch {e+1:2}/{len(epochs)}   {d_rounds:4d}×D {g_rounds:4d}×G » real {level_real:3.0%} vs. fake {level_fake:3.0%}")
    torch.save({
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
    }, f"facegen{e+1:03}.pth")
