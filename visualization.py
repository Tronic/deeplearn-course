import latent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sns.set_style("darkgrid")  # Make pyplot look better

class Base:
    def __init__(self, shape=()):
        self.shape = np.array(shape, dtype=np.int)
        self.z = latent.random(self.shape.prod())
    def __call__(self, **kwargs):
        pass

class Plot(Base):
    def __init__(self, rows=4, cols=4):
        super().__init__((rows, cols))

    def __call__(self, generator, discriminator):
        fig, axes = plt.subplots(*self.shape, figsize=2.0 * self.shape[::-1])
        fake = generator(self.z)
        levels = discriminator(fake).detach().cpu().numpy()
        levels = 1.0 / (1.0 + np.exp(-levels))  # Sigmoid for probability

        fake = fake.detach().cpu().permute(0, 2, 3, 1).numpy()
        for i, ax in enumerate(axes.flat):
            ax.imshow(fake[i] / 2 + .5)
            ax.grid(False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_title(f"{levels[i, 0]:.0%} real")
        plt.show()

class PNG(Base):
    def __init__(self, rows=4, cols=8):
        super().__init__((rows, cols))
        self.counter = 0

    def __call__(self, generator, **kwargs):
        s = 100  # Pixel-size of a single image
        fh, fw = s * self.shape  # Full height and width of collated image
        img = np.empty((fh, fw, 3), dtype=np.uint8)
        fake = generator(self.z)
        fake = ((fake.detach() + 1.0) * 127.5).cpu().permute(0, 2, 3, 1).numpy()
        fake = fake.reshape(*self.shape, s, s, 3)
        for r, c in np.ndindex(*self.shape):
            ir, ic = r * s, c * s
            img[ir : ir + s, ic : ic + s] = fake[r, c]
        Image.fromarray(img).save(f"training/img{self.counter:05}.png")
        self.counter += 1
