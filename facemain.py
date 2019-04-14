import facedata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

sns.set_style("darkgrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)

def images_tensor(images):
    """Scale, transpose and convert Numpy tensor into Torch tensor."""
    global device
    images = np.transpose(images, (0, 3, 2, 1))  # N, rgb, height, width
    return torch.Tensor(images, device=device) / 128.0 - 1.0  # -1 to 1

def ages_tensor(ages):
    return torch.tensor(ages.reshape(-1, 1).astype(np.float32), device=device)

#%% Initialize training
net = Net()
try:
    net.load_state_dict(torch.load("agenet2.pth"))
    print("Loaded agenet.pth")
except:
    print("Initialized agenet")
net.to(device)
criterion = nn.MSELoss()
training_N = facedata.N - 1000
stats = []

#%% Training epoch
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
minibatch_size = 64
minirounds = range(2)
rounds = range(training_N // minibatch_size)
print_stats = np.linspace(rounds[0], rounds[-1], 15, dtype=np.int)
losses = np.full(len(rounds), np.nan)
stats.append(losses)
for r in rounds:
    # Import a batch of data
    batch = r * minibatch_size + np.arange(minibatch_size)
    ages = ages_tensor(facedata.ages[batch])
    images = images_tensor(facedata.images[batch])
    batch = (batch + minibatch_size) % training_N
    # Optimize network
    for r2 in minirounds:
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, ages)
        loss.backward()
        optimizer.step()
    losses[r] = loss.item() ** .5
    # Statistics
    if r in print_stats:
        print(f"{r+1:4d}/{len(rounds)} {batch[0]:5d}/{training_N} » stddev {losses[r]:.0f} years")
        if r is not print_stats[0]:
            for ls in stats: plt.plot(ls)
            plt.ylim(0, 30)
            plt.show()

#%% Do a test run
cols, rows = 5, 4
batch = np.random.choice(range(training_N, facedata.N), cols * rows, replace=False)
with torch.no_grad():
    output = net(images_tensor(facedata.images[batch]))
    loss = criterion(output, ages_tensor(facedata.ages[batch]))
    print(f"Validation stddev {loss.item()**.5:.0f} years")

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
