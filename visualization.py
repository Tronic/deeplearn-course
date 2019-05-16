import latent
import numpy as np
import torch
from torch import nn
from PIL import Image

# For Video output
try:
    import subprocess
    from wand.color import Color
    from wand.image import Image
    from wand.drawing import Drawing
except:
    pass

class Base:
    def __init__(self, shape=(), device=None):
        self.shape = np.array(shape, dtype=np.int)
        self.z = latent.random(self.shape.prod(), device=device)
    def generate(self, generator, genargs):
        with torch.no_grad():
            fake = generator(self.z, **genargs)
            return ((fake + 1.0) * 127.5)
    def to_cpu(self, imgtensor):
        s = imgtensor.size(2)
        return imgtensor.permute(0, 2, 3, 1).view(*self.shape, s, s, 3).cpu().numpy()


class PNG(Base):
    def __init__(self, rows=4, cols=8, device=None):
        super().__init__((rows, cols), device=device)
        self.counter = 0

    def __call__(self, generator, discriminator, stats, **genargs):
        fake = self.to_cpu(self.generate(generator, genargs))
        s = fake.shape[2]
        fh, fw = s * self.shape  # Full height and width of collated image
        img = np.empty((fh, fw, 3), dtype=np.uint8)
        for r, c in np.ndindex(*self.shape):
            ir, ic = r * s, c * s
            img[ir : ir + s, ic : ic + s] = fake[r, c]
        Image.fromarray(img).save(f"training/img{self.counter:05}.png")
        self.counter += 1

class Video(Base):
    def __init__(self, rows=4, cols=8, device=None):
        super().__init__((rows, cols), device=device)

    def __enter__(self):
        self.ffmpeg = subprocess.Popen(
            'ffmpeg -framerate 60 -s 1280x660 -f rawvideo -pix_fmt rgb24 -i pipe: -c:v libx264 -crf 18 -y training.mkv'.split(),
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,

        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait(10)
        print("Saved to out.mkv")

    def __call__(self, generator, discriminator, stats, **genargs):
        fake = self.generate(generator, genargs)
        fake = nn.functional.interpolate(fake, 160)
        fake = self.to_cpu(fake)
        s = fake.shape[2]
        fh, fw = s * self.shape  # Full height and width of collated image
        assert fw == 1280 and fh == 640
        fh += 20
        img = np.empty((fh, fw, 3), dtype=np.uint8)
        for r, c in np.ndindex(*self.shape):
            ir, ic = r * s, c * s
            img[ir : ir + s, ic : ic + s] = fake[r, c]
        with Drawing() as draw:
            with Image(width=1280, height=20, background=Color("black")) as wimg:
                draw.font_family = "DejaVu Sans Mono"
                draw.font_size = 16.0
                draw.fill_color = Color("white")
                draw.text(20, 16, stats)
                draw(wimg)
                img[-20:, :, :] = np.array(wimg).reshape(20, 1280, 4)[:, :, :3]
        self.ffmpeg.stdin.write(img.tobytes())

