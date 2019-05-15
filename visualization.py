import latent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch

sns.set_style("darkgrid")  # Make pyplot look better

s = 50  # Pixel-size of a single image

class Base:
    def __init__(self, shape=(), device=None):
        self.shape = np.array(shape, dtype=np.int)
        self.z = latent.random(self.shape.prod(), device=device)
    def generate(self, generator):
        with torch.no_grad():
            fake = generator(self.z)
            return ((fake + 1.0) * 127.5)
    def to_cpu(self, imgtensor):
        return imgtensor.permute(0, 2, 3, 1).view(*self.shape, s, s, 3).cpu().numpy()


class Plot(Base):
    def __init__(self, rows=4, cols=4, device=None):
        super().__init__((rows, cols), device=device)

    def __call__(self, generator, discriminator):
        fig, axes = plt.subplots(*self.shape, figsize=2.0 * self.shape[::-1])
        with torch.no_grad():
            fake = self.generate(generator)
            levels = discriminator(fake).cpu().numpy()
        levels = 1.0 / (1.0 + np.exp(-levels))  # Sigmoid for probability

        fake = self.to_cpu(fake)
        for i, ax in enumerate(axes.flat):
            ax.imshow(fake[divmod(i, self.shape[0])] / 2 + .5)
            ax.grid(False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_title(f"{levels[i, 0]:.0%} real")
        plt.show()

class PNG(Base):
    def __init__(self, rows=4, cols=8, device=None):
        super().__init__((rows, cols), device=device)
        self.counter = 0

    def __call__(self, generator, **kwargs):
        fh, fw = s * self.shape  # Full height and width of collated image
        img = np.empty((fh, fw, 3), dtype=np.uint8)
        fake = self.to_cpu(self.generate(generator))
        for r, c in np.ndindex(*self.shape):
            ir, ic = r * s, c * s
            img[ir : ir + s, ic : ic + s] = fake[r, c]
        Image.fromarray(img).save(f"training/img{self.counter:05}.png")
        self.counter += 1
