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
            nn.Sigmoid()
        )

    def forward(self, faces):
        return self.seq(faces)


class Generator(nn.Module):
    """Compose 200x200x3 face images out of random numbers."""
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(100, 1024 * 4 * 4),
            nn.BatchNorm1d(1024 * 4 * 4),
            Reshape(1024, 4, 4),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)

#%% Init GAN
discriminator = Discriminator()
generator = Generator()

assert discriminator(torch.zeros(10, 3, 100, 100)).shape == torch.Size((10, 1))
assert generator(torch.zeros(10, 100)).shape == torch.Size((10, 3, 100, 100))

generator.to(device)
discriminator.to(device)

# net.load_state_dict(torch.load("agenet.pth"))

criterion = nn.BCELoss()
stats = []

#%% Training epochs
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
minibatch_size = 32
rounds = range(facedata.N // minibatch_size // 100)
epochs = range(20 if device.type == "cuda" else 2)
targets_real = torch.ones((minibatch_size, 1), device=device)
targets_fake = torch.zeros((minibatch_size, 1), device=device)
targets = torch.cat((targets_real, targets_fake), dim=0)

for e in epochs:
    stats.append(np.empty(len(rounds)))
    for r in rounds:
        # Train the discriminator
        batch = slice(r * minibatch_size, (r + 1) * minibatch_size)
        real = images[batch].to(torch.float32) / 255.0
        with torch.no_grad():
            fake = generator(torch.randn((minibatch_size, 100), device=device))
        d_optimizer.zero_grad()
        output = discriminator(torch.cat((real, fake), dim=0))
        loss = criterion(output, targets)
        loss.backward()
        d_optimizer.step()
        stats[-1][r] = loss.item()
        # Train the generator
        g_optimizer.zero_grad()
        fake = generator(torch.randn((minibatch_size, 100), device=device))
        output = discriminator(fake)
        loss = criterion(output, targets_real)
        loss.backward()
        g_optimizer.step()

    # Statistics
    print(f"Epoch {e+1:2d}/{len(epochs)} » loss {stats[-1].mean():.2f}")
    for losses in stats: plt.plot(losses)
    plt.show(block=False)

#%% Do a test run
cols, rows = 5, 4
with torch.no_grad():
    fake = generator(torch.randn((cols * rows, 100), device=device))
    output = discriminator(fake)

# Show the results
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake[i].cpu().transpose(0, 2))
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(f"{output[i, 0]:.0%} real")

plt.show()

#%% Save current state
torch.save(net.state_dict(), "agenet.pth"); print("Saved to agenet.pth")
