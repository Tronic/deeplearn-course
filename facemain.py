import facedata
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class Linearize(nn.Module):
    """Convert four-dimensional image tensor into one suitable for linear layers."""
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    """A simple convolutional network for estimating person's age."""
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Linearize(),
            nn.Linear(8 * 23 * 23, 1)
        )

    def forward(self, x):
        return self.seq(x)

def images_tensor(images):
    """Scale, transpose and convert Numpy tensor into Torch tensor."""
    images = np.transpose(images, (0, 3, 2, 1))  # N, rgb, height, width
    return torch.Tensor(images) / 128.0 - 1.0  # -1 to 1

# Initialize for learning
net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
rounds = range(100)
losses = np.zeros(len(rounds))

for r in rounds:
    # Import a batch of data
    batch = np.random.choice(facedata.N - 100, 32)
    ages = torch.tensor(facedata.ages[batch, np.newaxis], dtype=torch.float)
    images = images_tensor(facedata.images[batch])
    # Optimize network
    optimizer.zero_grad()
    output = net(images)
    loss = criterion(output, ages)
    loss.backward()
    optimizer.step()
    # Statistics
    losses[r] = loss.item()
    if r in np.linspace(rounds[0], rounds[-1], 10, dtype=np.int):
        print(f"{r:4d}/{rounds[-1]} loss={loss.item():.0f}")
        plt.plot(losses)
        plt.ylim(0, 1000)
        plt.show()

# Do a test run
batch = facedata.N - 100 + np.random.choice(100, 5 * 4)
output = net(images_tensor(facedata.images[batch]))
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
for i, idx in enumerate(batch):
    out = output[i, 0].item()
    image = facedata.images[idx]
    age = facedata.ages[idx]
    gender = "female" if facedata.genders[idx] else "male"
    race = facedata.Race(facedata.races[idx])

    ax = axes.flat[i]
    ax.imshow(facedata.images[idx])
    ax.set_title(f"out={out:.0f} for {age} year old {race.name} {gender}")

plt.show()
