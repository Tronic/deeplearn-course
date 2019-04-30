import facedata
import numpy as np
import torch
import torch.nn as nn
import time

#%% Setup

def images_tensor(images, stride=2):
    """Scale, transpose and convert Numpy tensor into Torch tensor."""
    global device
    images = np.transpose(images, (0, 3, 1, 2))[:, :, ::stride, ::stride]  # N, rgb, height, width
    return torch.tensor(images, device=device, dtype=torch.uint8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Uploading tensors to", device)
images = images_tensor(facedata.images)

assert images.shape == torch.Size((facedata.N, 3, 100, 100))

# How many random numbers are fed to generator
zdim = 100

def latent_tensor(*size):
    """Create random latent variables for size samples."""
    global zdim, device
    z = torch.randn((*size, zdim), device=device)
    lengths = (z * z).sum(dim=-1, keepdim=True)
    return z / lengths * zdim  # Form a hypersphere of radius zdim


#%% Visualization init

visualization = "png"
cols, rows = 4, 4

if visualization == "plt":
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("darkgrid")  # Make pyplot look better

    def visualize():
        global cols, rows, z_test, generator
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        fake_test = generator(z_test)
        levels_test = discriminator(fake_test).detach().cpu().numpy()
        levels_test = 1.0 / (1.0 + np.exp(-levels_test))  # Sigmoid for probability

        fake_test = fake_test.detach().cpu().permute(0, 2, 3, 1).numpy()
        for i, ax in enumerate(axes.flat):
            ax.imshow(fake_test[i] / 2 + .5)
            ax.grid(False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_title(f"{levels_test[i, 0]:.0%} real")
        plt.show()
elif visualization == "png":
    from PIL import Image
    cols, rows = 8, 5
    counter = 0
    def visualize():
        global cols, rows, z_test, generator, counter
        s = 100
        fake_test = ((generator(z_test).detach() + 1.0) * 127.5).cpu().permute(0, 2, 3, 1).numpy()
        img = np.empty((rows * s, cols * s, 3), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                img[r * s : (r + 1) * s, c * s : (c + 1) * s, :] = fake_test[r * cols + c]
        Image.fromarray(img).save(f"training/img{counter:05}.png")
        counter += 1

else:
    def visualize():
        pass

z_test = latent_tensor(rows * cols)

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
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1, inplace=True),
            Reshape(-1),
            nn.Linear(64 * 12 * 12, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, faces):
        return self.seq(faces)

class Generator(nn.Module):
    """Convolutional generator adapted from DCGAN by Radford et al."""
    def __init__(self):
        global zdim
        super().__init__()
        CH = 1024
        self.seq = nn.Sequential(
            nn.Linear(zdim, CH * 4 * 4, bias=False),
            Reshape(CH, 4, 4),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=CH, out_channels=CH // 2, kernel_size=5, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(CH // 2),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=CH // 2, out_channels=CH // 4, kernel_size=5, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(CH // 4),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=CH // 4, out_channels=CH // 4, kernel_size=5, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(CH // 4),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=CH // 4, out_channels=CH // 4, kernel_size=5, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(CH // 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(in_channels=CH // 4, out_channels=CH // 4, kernel_size=5, padding=2, stride=2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(in_channels=CH // 4, out_channels=3, kernel_size=1, bias=True),  # Map channels to colors
            nn.Tanh()
        )

    def forward(self, x):
        x = self.seq(x)
        assert x.size(-1) == 97, f"got {x.size(-1)} pixels wide"
        return nn.functional.interpolate(x, scale_factor=100/x.size(-1) + 1e-5, mode="bilinear", align_corners=False)

g_output = Generator()(torch.zeros(10, zdim))
assert g_output.shape == torch.Size((10, 3, 100, 100)), f"Generator output {g_output.shape}"
d_output = Discriminator()(g_output)
assert d_output.shape == torch.Size((10, 1)), f"Discriminator output {d_output.shape}"
del g_output, d_output


#%% Init GAN
discriminator = Discriminator()
generator = Generator()

try:
    discriminator.load_state_dict(torch.load("discriminator.pth", map_location=lambda storage, location: storage))
    generator.load_state_dict(torch.load("generator.pth", map_location=lambda storage, location: storage))
    print("Networks loaded")
except:
    pass

generator.to(device)
discriminator.to(device)

#%% Training epochs
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(.5, .999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.00005, betas=(.5, .998))
criterion = nn.BCEWithLogitsLoss()
minibatch_size = 64
rounds = range(facedata.N // minibatch_size if device.type == "cuda" else 10)
epochs = range(60 * 300)
ones = torch.ones((minibatch_size, 1), device=device)
zeros = torch.zeros((minibatch_size, 1), device=device)

print(f"Training with {len(rounds)} rounds per epoch:")
for e in epochs:
    d_rounds = g_rounds = 0
    rtimer = time.perf_counter()
    for r in rounds:
        print(f"  [{'*' * (25 * r // rounds[-1]):25s}] {r+1:4d}/{len(rounds)} {(time.perf_counter() - rtimer) / (r + .1) * len(rounds):3.0f} s/epoch", end="\r")
        # Choose a random slice of images
        batch = np.random.randint(facedata.N - minibatch_size)
        batch = slice(batch, batch + minibatch_size)
        # Run the discriminator on real and fake data
        real = images[batch].to(torch.float32)
        real /= 128.0
        real -= 1.0
        d_optimizer.zero_grad()
        z = latent_tensor(minibatch_size)
        fake = generator(z)
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
        # Train the discriminator only if it is not too good
        if level_diff < 0.6:
            criterion(output_real, ones).backward()
            criterion(output_fake, zeros).backward()
            d_optimizer.step()
            d_rounds += 1
        # Train the generator only if the discriminator works
        if level_diff > 0.4:
            g_optimizer.zero_grad()
            criterion(discriminator(fake), ones).backward()
            g_optimizer.step()
            g_rounds += 1
            visualize()
    print(f"  Epoch {e+1:2d}/{len(epochs)}   {d_rounds:4d}×D {g_rounds:4d}×G » real {level_real:3.0%} vs. fake {level_fake:3.0%}")
    # Show the results
    #visualize()

#%% Save current state
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
print("Saved to generator.pth and discriminator.pth!")
