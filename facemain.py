import facedata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

#%% Setup

sns.set_style("darkgrid")  # Make pyplot look better

def images_tensor(images):
    """Scale, transpose and convert Numpy tensor into Torch tensor."""
    global device
    images = np.transpose(images, (0, 3, 1, 2))  # N, rgb, height, width
    return torch.tensor(images, device=device, dtype=torch.uint8)

def ages_tensor(ages):
    global device
    return torch.tensor(ages.reshape(-1, 1).astype(np.float32), device=device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Uploading tensors to", device)
ages = ages_tensor(facedata.ages)
images = images_tensor(facedata.images)

#%% Network definition

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
            nn.LeakyReLU(0.1),
            Linearize(),
            nn.Linear(8 * 23 * 23, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.seq(x.to(torch.float32) / 128.0 - 1.0)

#%% Initialize training
net = Net()
try:
    net.load_state_dict(torch.load("agenet.pth"))
    print("Loaded agenet.pth")
except:
    print("Initialized agenet")
net.to(device)
criterion = nn.MSELoss()
training_N = facedata.N - 1000
stats = []

#%% Training epochs
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
minibatch_size = 32
rounds = range(training_N // minibatch_size)
epochs = range(20)

for e in epochs:
    stats.append(np.empty(len(rounds)))
    for r in rounds:
        batch = slice(r * minibatch_size, (r + 1) * minibatch_size)
        # Optimize network for minibatch
        optimizer.zero_grad()
        output = net(images[batch])
        loss = criterion(output, ages[batch])
        loss.backward()
        optimizer.step()
        stats[-1][r] = loss.item() ** .5

    # Statistics
    print(f"Epoch {e+1:2d}/{len(epochs)} » stddev {stats[-1].mean():.1f} years")
    for losses in stats: plt.plot(losses)
    plt.ylim(0, 20)
    plt.show(block=False)

#%% Do a test run
cols, rows = 5, 4
batch = np.random.choice(range(training_N, facedata.N), cols * rows, replace=False)
with torch.no_grad():
    output = net(images[batch])
    loss = criterion(output, ages[batch])
    print(f"Validation stddev {loss.item()**.5:.1f} years")

# Show the results
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
for i, idx in enumerate(batch):
    out = output[i, 0].item()
    image = facedata.images[idx]
    age = facedata.ages[idx]
    gender = "female" if facedata.genders[idx] else "male"
    race = facedata.Race(facedata.races[idx])

    ax = axes.flat[i]
    ax.imshow(facedata.images[idx])
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(f"{out:.0f} ~ {age} year {race.name} {gender}")

plt.show()

#%% Save current state
torch.save(net.state_dict(), "agenet.pth"); print("Saved to agenet.pth")
