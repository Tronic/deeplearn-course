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
    """Convolutional generator adapted from DCGAN by Radford et al."""
    def __init__(self):
        super().__init__()
        CH = 512
        self.seq = nn.Sequential(
            nn.Linear(100, CH * 4 * 4),
            nn.BatchNorm1d(CH * 4 * 4),
            Reshape(CH, 4, 4),
            nn.ConvTranspose2d(in_channels=CH, out_channels=CH // 2, kernel_size=4, stride=2),
            nn.BatchNorm2d(CH // 2),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=CH // 2, out_channels=CH // 4, kernel_size=4, stride=2),
            nn.BatchNorm2d(CH // 4),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=CH // 4, out_channels=CH // 8, kernel_size=4, stride=2, dilation=2),
            nn.BatchNorm2d(CH // 8),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=CH // 8, out_channels=3, kernel_size=4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)

#%% Init GAN
discriminator = Discriminator()
generator = Generator()

assert discriminator(torch.zeros(10, 3, 100, 100)).shape == torch.Size((10, 1))
assert generator(torch.zeros(10, 100)).shape == torch.Size((10, 3, 100, 100))

try:
    discriminator.load_state_dict(torch.load("discriminator.pth"))
    generator.load_state_dict(torch.load("generator.pth"))
    print("Networks loaded")
except:
    pass

generator.to(device)
discriminator.to(device)

#%% Training epochs
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, weight_decay=1e-3)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
criterion = nn.BCELoss()
minibatch_size = 16
rounds = range(facedata.N // minibatch_size)
epochs = range(5)
targets_real = torch.ones((minibatch_size, 1), device=device)
targets_fake = torch.zeros((minibatch_size, 1), device=device)
targets = torch.cat((targets_real, targets_fake), dim=0)

print(f"Training with {len(rounds)} rounds per epoch:")
for e in epochs:
    d_rounds = g_rounds = 0
    for r in rounds:
        # Run the discriminator on real and fake data
        batch = slice(r * minibatch_size, (r + 1) * minibatch_size)
        real = images[batch].to(torch.float32) / 255.0
        with torch.no_grad():
            fake = generator(torch.randn((minibatch_size, 100), device=device))
        d_optimizer.zero_grad()
        data = torch.cat((real, fake), dim=0)
        output = discriminator(data)
        # Check levels
        realout = output[:minibatch_size].mean().item()
        fakeout = output[minibatch_size:].mean().item()
        # Train the discriminator only if it is not too good
        if realout - fakeout < 0.6:
            loss = criterion(output, targets)
            loss.backward()
            d_optimizer.step()
            d_rounds += 1
        # Train the generator only if the discriminator works
        if realout - fakeout > 0.4:
            g_optimizer.zero_grad()
            fake = generator(torch.randn((minibatch_size, 100), device=device))
            loss = criterion(discriminator(fake), targets_real)
            loss.backward()
            g_optimizer.step()
            g_rounds += 1
    print(f"  Epoch {e+1:2d}/{len(epochs)}   {d_rounds:4d}×D {g_rounds:4d}×G » real {realout:3.0%} vs. fake {fakeout:3.0%}")
    # Show the results
    cols, rows = 4, 8  # 2 * minibatch_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].cpu().permute(1, 2, 0))
        ax.grid(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title(f"{output[i, 0]:.0%} real")
    plt.show()

#%% Save current state
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

