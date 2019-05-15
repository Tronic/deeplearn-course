import facedata
import numpy as np
import torch
import torch.nn as nn
import time
import latent
import visualization

#%% Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 100
print("Uploading tensors to", device)
images = facedata.torch_tensor(stride=2, device=device)
assert images.shape == torch.Size((facedata.N, 3, image_size, image_size))

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
            nn.LeakyReLU(0.25, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.Dropout2d(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.25, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.25, inplace=True),
#            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
#            nn.Dropout2d(0.2, inplace=True),
#            nn.MaxPool2d(2),
#            nn.LeakyReLU(0.25, inplace=True),
#            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
#            nn.LeakyReLU(0.25, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.Dropout2d(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.25, inplace=True),
            Reshape(-1),
            nn.Linear(40_000, 256),
            nn.LeakyReLU(0.25, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.25, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.25, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, faces):
        # Calculate as sum of normal and horizontally mirrored faces
        return self.seq(faces) + self.seq(faces.flip(dims=(3,)))

class UpConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2, bias=True),
            nn.Tanh(),
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.seq(x)


class LatentIn(nn.Module):
    def __init__(self, image_size, channels=latent.dimension):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.seq = nn.Sequential(
            nn.Linear(latent.dimension, channels - 2, bias=False),
            nn.LeakyReLU(0.125, inplace=True),
        )
        ls = torch.linspace(-1.0, 1.0, image_size)
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
            nn.Sigmoid(),
            nn.ConvTranspose2d(in_channels=ch, out_channels=3, kernel_size=1),
            nn.Tanh())
    def forward(self, x):
        x = self.seq(x)
        return nn.functional.interpolate(x, image_size, mode="bilinear", align_corners=False)

class Generator(nn.Module):
    """Convolutional generator adapted from DCGAN by Radford et al."""
    def __init__(self):
        super().__init__()
        CH, SIZE = 256, 5  # Initial channel count and image resolution
        self.init_size = SIZE
        self.latimg = nn.Linear(latent.dimension, SIZE * SIZE, bias=False)
        self.lat = LatentIn(image_size=SIZE << 5, channels=CH)
        self.upc = nn.ModuleList([
            UpConvLayer(CH + 1, CH),
            UpConvLayer(2 * CH, CH),
            UpConvLayer(2 * CH, CH),
            UpConvLayer(2 * CH, CH),
            UpConvLayer(2 * CH, CH),
        ])
        # Map channels to colors
        self.toRGB = ToRGB(CH)

    def forward(self, latent, train=False):
        x = 256.0 * self.latimg(latent).view(-1, 1, self.init_size, self.init_size)
        latent = self.lat(latent)
        for l in range(4):
            if train: abs(0.5 - x.std(dim=0).mean()).backward(retain_graph=True)
            lat = nn.functional.interpolate(latent, x.size(2))
            x = self.upc[l](torch.cat((x, lat), dim=1))
        return self.toRGB(x)

# Verify network output shapes
g_output = Generator()(latent.random(10))
assert g_output.shape == torch.Size((10, 3, image_size, image_size)), f"Generator output {g_output.shape}"
d_output = Discriminator()(g_output)
assert d_output.shape == torch.Size((10, 1)), f"Discriminator output {d_output.shape}"
del g_output, d_output


#%% Init GAN
visualize = visualization.PNG(device=device)

discriminator = Discriminator()
generator = Generator()

generator.to(device)
discriminator.to(device)

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(.5, .99999))
g_optimizer = torch.optim.Adam([
    {"params": generator.latimg.parameters(), "lr": 0.001},
    {"params": generator.lat.parameters(), "lr": 0.0005},
    {"params": generator.upc.parameters(), "lr": 0.00002},
    {"params": generator.toRGB.parameters(), "lr": 0.001},
], betas=(.8, .99999))

criterion = nn.BCEWithLogitsLoss()

#%% Load previous state
try:
    from glob import glob
    filename = glob("facegen*.pth")[-1]
    checkpoint = torch.load(filename, map_location=lambda storage, location: storage)
    discriminator.load_state_dict(checkpoint["discriminator"])
    generator.load_state_dict(checkpoint["generator"])
    print("Networks loaded:", filename)
except:
    pass


#%% Training
minibatch_size = 32
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
        print(f"  [{'*' * (25 * r // rounds[-1]):25s}] {g_rounds:03d}:{d_rounds:03d}  {(time.perf_counter() - rtimer) / (r + .1) * len(rounds):3.0f} s/epoch {level_real:4.0%} vs{level_fake:4.0%}", end="\N{ESC}[K\r")
        # Make a set of fakes
        z = latent.random(minibatch_size, device=device)
        g_optimizer.zero_grad()
        fake = generator(z, train=True)
        # Train the generator
        #var = fake.var(dim=0).mean()  # Variance across samples
        loss = criterion(discriminator(fake), ones)
        loss.backward()
        g_optimizer.step()
        g_rounds += 1
        # Train the discriminator one time or until it is good enough
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
            # Train the discriminator
            criterion(output_real, ones).backward()
            criterion(output_fake, zeros).backward()
            d_optimizer.step()
            d_rounds += 1
            # Check levels
            levels = np.stack((
                output_real.detach().cpu().numpy().reshape(minibatch_size),
                output_fake.detach().cpu().numpy().reshape(minibatch_size)
            ))
            levels = 1.0 / (1.0 + np.exp(-levels))  # Sigmoid for probability
            level_real, level_fake = levels.mean(axis=1)
            level_diff = level_real - level_fake
            if level_diff > 0.2: break  # Good enough
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = 0.0001
        if level_diff > 0.7:
            for param_group in d_optimizer.param_groups:
                param_group['lr'] *= 0.9
        visualize(generator=generator, discriminator=discriminator)
    print(f"  Epoch {e+1:2}/{len(epochs)} {g_rounds:4d}×G {d_rounds:4d}×D » real{level_real:4.0%} vs. fake{level_fake:4.0%}", end="\N{ESC}[K\n")
    torch.save({
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
    }, f"facegen{e+1:03}.pth")
