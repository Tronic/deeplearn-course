import facedata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

#%% Setup

sns.set_style("darkgrid")  # Make pyplot look better

def images_tensor(images, stride=2):
    """Scale, transpose and convert Numpy tensor into Torch tensor."""
    global device
    images = np.transpose(images, (0, 3, 1, 2))[:, :, ::stride, ::stride]  # N, rgb, height, width
    return torch.tensor(images, device=device, dtype=torch.uint8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Uploading tensors to", device)
images = images_tensor(facedata.images)

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
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2),
            Reshape(-1),
            nn.Linear(8 * 23 * 23, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, faces):
        return self.seq(faces)

# How many random numbers are fed to generator
zdim = 10

class Generator(nn.Module):
    """Convolutional generator adapted from DCGAN by Radford et al."""
    def __init__(self):
        global zdim
        super().__init__()
        CH = 512
        self.seq = nn.Sequential(
            nn.Linear(zdim, CH * 4 * 4, bias=False),
            nn.BatchNorm1d(CH * 4 * 4),
            Reshape(CH, 4, 4),
            nn.ConvTranspose2d(in_channels=CH, out_channels=CH // 2, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(CH // 2),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=CH // 2, out_channels=CH // 4, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(CH // 4),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=CH // 4, out_channels=CH // 8, kernel_size=4, stride=2, dilation=2, bias=False),
            nn.BatchNorm2d(CH // 8),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=CH // 8, out_channels=3, kernel_size=4, stride=2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)

#%% Init GAN
discriminator = Discriminator()
generator = Generator()

g_output = generator(torch.zeros(10, zdim))
d_output = discriminator(g_output)
assert g_output.shape == torch.Size((10, 3, 100, 100)), f"Generator output {g_output.shape}"
assert d_output.shape == torch.Size((10, 1)), f"Discriminator output {d_output.shape}"
del g_output, d_output

try:
    discriminator.load_state_dict(torch.load("discriminator.pth"))
    generator.load_state_dict(torch.load("generator.pth"))
    print("Networks loaded")
except:
    pass

generator.to(device)
discriminator.to(device)

#%% Training epochs
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(.5, .999), weight_decay=1e-3)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(.5, .999))
criterion = nn.BCEWithLogitsLoss()
minibatch_size = 16
rounds = range(facedata.N // minibatch_size)
epochs = range(50)
targets_real = torch.ones((minibatch_size, 1), device=device)
targets_fake = torch.zeros((minibatch_size, 1), device=device)
targets = torch.cat((targets_real, targets_fake), dim=0)

import time

class Timer:
    def __init__(self):
        self.name = None
        self.stats = {}
    def __call__(self, name):
        t = time.perf_counter()
        if self.name:
            self.stats[self.name] = self.stats.get(self.name, 0.0) + (t - self.t)
        self.name = name
        self.t = t
    def dump(self):
        print("Time spent:")
        for n, t in self.stats.items():
            print(f"{n:14s}{t:7.2f}")

print(f"Training with {len(rounds)} rounds per epoch:")
timer = Timer()
for e in epochs:
    d_rounds = g_rounds = 0
    for r in rounds:
        print(f"  [{'*' * (30 * r // rounds[-1]):30s}] {r+1:4d}/{len(rounds)}", end="\r")
        # Run the discriminator on real and fake data
        timer("batch load")
        batch = slice(r * minibatch_size, (r + 1) * minibatch_size)
        real = images[batch].to(torch.float32)
        real /= 255.0
        timer("update init")
        d_optimizer.zero_grad()
        timer("random")
        z = torch.randn((minibatch_size, zdim), device=device)
        timer("fake")
        with torch.no_grad():
            fake = generator(z)
        output_fake = discriminator(fake)
        timer("real")
        output_real = discriminator(real)
        # Check levels (only every few rounds because it is slow)
        if r % 10 == 0 or r == rounds[-1]:
            timer("levels")
            realout = output_real.detach().cpu().numpy().reshape(minibatch_size)
            fakeout = output_fake.detach().cpu().numpy().reshape(minibatch_size)
            out = 1 / (1 + np.exp(-np.array([realout, fakeout])))
            realout, fakeout = out.mean(axis=1)
            diff = realout - fakeout
        # Train the discriminator only if it is not too good
        if diff < 0.8:
            timer("d update")
            criterion(output_real, targets_real).backward()
            criterion(output_fake, targets_fake).backward()
            d_optimizer.step()
            d_rounds += 1
        # Train the generator only if the discriminator works
        if diff > 0.2:
            timer("g update")
            g_optimizer.zero_grad()
            fake = generator(z)
            output_fake = discriminator(fake)
            criterion(output_fake, targets_real).backward()
            g_optimizer.step()
            g_rounds += 1
        timer(None)
    print(f"  Epoch {e+1:2d}/{len(epochs)}   {d_rounds:4d}×D {g_rounds:4d}×G » real {realout:3.0%} vs. fake {fakeout:3.0%}")
    # Show the results
    cols, rows = 4, 4  # minibatch_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake[i].detach().cpu().permute(1, 2, 0))
        ax.grid(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title(f"{out[1, i]:.0%} real")
    plt.show()

timer.dump()

#%% Save current state
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
print("Saved to generator.pth and discriminator.pth!")
