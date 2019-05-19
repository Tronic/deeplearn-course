import facedata
import torch
import torch.nn as nn
import time
import latent
import visualization
from networks import Generator, Discriminator, base_size, max_size

#%% Setup

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
faces = facedata.Torch(device=device)

# Training settings
epochs = range(6)
rounds = range(3000 if device.type == "cuda" else 10)
minibatch_size = 16

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
def training():
    print(f"Training with {len(rounds)} rounds per epoch:")
    ones = torch.ones((minibatch_size, 1), device=device)
    zeros = torch.zeros((minibatch_size, 1), device=device)
    criterion = nn.BCEWithLogitsLoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(.5, .999))
    g_optimizer = torch.optim.Adam([
        {"params": generator.latimg.parameters(), "lr": 0.0005},
        {"params": generator.inject.parameters(), "lr": 0.0001},
        {"params": generator.upc.parameters(), "lr": 0.00003},
        {"params": generator.toRGB.parameters(), "lr": 0.00005},
    ], betas=(.8, .99))
    image_size = base_size
    noise = alpha = 0.0
    for e in epochs:
        if image_size < max_size:
            image_size *= 2
            alpha = 0.0
        images = faces.batch_iter(minibatch_size, image_size=image_size)
        rtimer = time.perf_counter()
        for r in rounds:
            alpha = min(1.0, alpha + 3 / len(rounds))  # Alpha blending after switching to bigger resolution
            # Make a set of fakes
            z = latent.random(minibatch_size, device=device)
            g_optimizer.zero_grad()
            fake = generator(z, image_size=image_size, alpha=alpha, diversify=10.0)
            # Train the generator
            criterion(discriminator(fake, alpha), ones).backward()
            g_optimizer.step()
            # Prepare images for discriminator training
            real = next(images)
            fake = fake.detach()  # Drop gradients (we don't want more generator updates)
            assert real.shape == fake.shape
            if noise:
                real += noise * torch.randn_like(real)
                fake += noise * torch.randn_like(fake)
            # Train the discriminator
            d_optimizer.zero_grad()
            output_fake = discriminator(fake, alpha)
            output_real = discriminator(real, alpha)
            criterion(output_real, ones).backward()
            criterion(output_fake, zeros).backward()
            d_optimizer.step()
            # Check levels and adjust noise
            level_real, level_fake = torch.sigmoid(torch.stack([
                output_real.detach(),
                output_fake.detach()
            ])).view(2, minibatch_size).mean(dim=1).cpu().numpy()
            level_diff = level_real - level_fake
            if level_diff > 0.95: noise += 0.001
            elif level_diff < 0.5: noise = max(0.0, noise - 0.001)
            # Status & visualization
            glr = g_optimizer.param_groups[0]['lr']
            bar = f"E{e} {image_size:3}px [{'»' * (15 * r // rounds[-1]):15s}] {r+1:04d}"
            alp = 2 * " ░▒▓█"[int(alpha * 4)]
            stats = f"α={alp} lr={glr*1e6:03.0f}µ ε={noise*1e3:03.0f}m {level_real:4.0%} vs{level_fake:4.0%}  {(time.perf_counter() - rtimer) / (r + .1) * len(rounds):3.0f} s/epoch"
            if (r+1) % 5 == 0: visualize(f"{bar} {stats}", image_size=image_size, alpha=alpha)
            print(f"\r  {bar} {stats} ", end="\N{ESC}[K")
        # After an epoch is finished:
        print()
        torch.save({
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        }, f"facegen{e:03}.pth")
        # Reduce learning rates
        for pg in g_optimizer.param_groups: pg['lr'] *= 0.8
        for pg in d_optimizer.param_groups: pg['lr'] *= 0.6

if __name__ == "__main__":
    with visualization.Video(generator, device=device) as visualize:
        training()
