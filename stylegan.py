import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
import torch.utils
from torch.nn import functional as F
from torchvision import transforms, utils, datasets
import time
import random
import math
import subprocess
import os

# Stylegan, code based on implementation from (https://github.com/rosinality/style-based-gan-pytorch)

#%% Setup


def get_git_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode("utf-8").strip()


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# git hash is used to identify the revision we calculated outputs with
git_hash = get_git_hash()
print("Training with {}".format(git_hash))
sample_dir = f"./sample/{git_hash}"
checkpoint_dir = f"./checkpoint/{git_hash}"

make_dir(sample_dir)
make_dir(checkpoint_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_data(dataset, batch_size, image_size=4):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset.transform = transform
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=16)

    return loader

# ------------- Network definition -------------


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = equal_lr(linear)

    def forward(self, x):
        return self.linear(x)


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.normal_()
        self.conv = equal_lr(conv)

    def forward(self, x):
        x = self.conv(x)
        return x


class AdaIn(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel*2)
        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, x, latent):
        style = self.style(latent)
        style = style.unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        x = self.norm(x)
        x = gamma * x + beta
        return x


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)


class Blur(nn.Module):
    def __init__(self):
        super(Blur, self).__init__()
        blur_kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        weight = torch.tensor(blur_kernel, dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        # Save as non-optimizable parameter
        self.register_buffer('weight', weight)

    def forward(self, x):
        out = F.conv2d(
                x,
                self.weight.repeat(x.shape[1], 1, 1, 1),
                padding=1,
                groups=x.shape[1])
        return out


class ConvBlock(nn.Module):
    """A two-layer convolutional block for the discriminator."""
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 padding,
                 kernel_size2=None,
                 padding2=None):
        super().__init__()
        pad1 = padding
        pad2 = padding2 if padding2 is not None else padding

        kernel1 = kernel_size
        kernel2 = kernel_size2 if kernel_size2 is not None else kernel_size

        self.conv = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
            EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConstantInput(nn.Module):
    """Random data tensor initialized on module init, stays constant for every sample."""
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, x):
        batch = x.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class StyledConvBlock(nn.Module):
    """Convolutional block of the stylegan generator."""
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,
                 padding=1,
                 style_dim=512,
                 initial=False):
        super().__init__()
        if initial:
            self.conv1 = ConstantInput(in_channel)
        else:
            self.conv1 = EqualConv2d(in_channel, out_channel, kernel_size, padding=padding)

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaIn(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaIn(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise):
        x = self.conv1(x)
        x = self.noise1(x, noise)
        x = self.adain1(x, style)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.noise2(x, noise)
        x = self.adain2(x, style)
        x = self.lrelu2(x)

        return x


class Generator(nn.Module):
    """Synthesis network"""
    def __init__(self):
        super().__init__()
        self.progression = nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True),
                StyledConvBlock(512, 512, 3, 1),
                StyledConvBlock(512, 512, 3, 1),
                StyledConvBlock(512, 512, 3, 1),
                StyledConvBlock(512, 256, 3, 1),
                StyledConvBlock(256, 128, 3, 1),
                StyledConvBlock(128, 64, 3, 1),
                StyledConvBlock(64, 32, 3, 1),
                StyledConvBlock(32, 16, 3, 1),
            ]
        )
        self.to_rgb = nn.ModuleList(
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                EqualConv2d(32, 3, 1),
                EqualConv2d(16, 3, 1),

            ]
        )

    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0]
        if len(style) < 2:
            inject_index = [len(self.progression) + 1]
        else:
            inject_index = random.sample(list(range(step)), len(style)-1)

        crossover = 0
        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))
                style_step = style[crossover]
            elif mixing_range[0] <= i <= mixing_range[1]:
                style_step = style[0]
            else:
                style_step = style[1]

            if i > 0 and step > 0:
                upsample = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
                # print(len(noise), i)
                out = conv(upsample, style_step, noise[i])
            else:
                out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)
                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i-1](upsample)
                    out = (1 - alpha) * skip_rgb + alpha * out
                break

        return out


class StyledGenerator(nn.Module):
    """Full stylegan generator, including latent mappings."""
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()
        self.generator = Generator()
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self,
                x,
                noise=None,
                step=0,
                alpha=-1,
                mean_style=None,
                style_weight=0,
                mixing_range=(-1, -1)):
        styles = []
        if type(x) not in (list, tuple):
            x = [x]

        for i in x:
            styles.append(self.style(i))

        batch = x[0].shape[0]
        if noise is None:
            noise = []
            for i in range(step+1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size, device=device))

        if mean_style is not None:
            styles_norm = []
            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))
            styles = styles_norm

        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range)

    def mean_style(self, x):
        style = self.style(x).mean(0, keepdim=True)
        return style


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1),
                ConvBlock(32, 64, 3, 1),
                ConvBlock(64, 128, 3, 1),
                ConvBlock(128, 256, 3, 1),
                ConvBlock(256, 512, 3, 1),
                ConvBlock(512, 512, 3, 1),
                ConvBlock(512, 512, 3, 1),
                ConvBlock(512, 512, 3, 1),
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )

        self.from_rgb = nn.ModuleList(
            [
                EqualConv2d(3, 16, 1),
                EqualConv2d(3, 32, 1),
                EqualConv2d(3, 64, 1),
                EqualConv2d(3, 128, 1),
                EqualConv2d(3, 256, 1),
                EqualConv2d(3, 512, 1),
                EqualConv2d(3, 512, 1),
                EqualConv2d(3, 512, 1),
                EqualConv2d(3, 512, 1),
            ]
        )

        self.n_layer = len(self.progression)
        self.linear = EqualLinear(512, 1)

    def forward(self, x, step=0, alpha=-1):
        for i in range(step, -1, -1):
            # print(x.shape)
            # print("Discriminator forward", step)
            index = self.n_layer - i - 1
            if i == step:
                out = self.from_rgb[index](x)
            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                # print(out.shape, mean_std.shape)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)
            if i > 0:
                out = F.interpolate(out, scale_factor=0.5, mode="bilinear", align_corners=False)
                if i == step and 0 <= alpha < 1:
                    skip_rgb = self.from_rgb[index+1](x)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=0.5, mode="bilinear", align_corners=False)
                    out = (1 - alpha) * skip_rgb + alpha * out

        # print(out.shape)
        out = out.squeeze(2).squeeze(2)
        # print(out.shape)
        out = self.linear(out)
        return out


# ------------- Training loop code -------------

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class TrainParams:
    def __init__(self):
        self.code_size = 512
#        self.batch_size = 16
        self.n_critic = 1
        self.phase = 64000
        self.lr_base = 0.001
        self.lr = {}
        self.init_size = 32
        self.max_size = 128
        self.mixing = True
        self.batch_size = {
            8: 16,
            16: 16,
            32: 16,
            64: 8,
            128: 8,
            256: 3
        }
        self.gen_sample = {512: (8, 4), 1024: (4, 2)}


def train(params: TrainParams,
          generator,
          discriminator):

    # The actual images need to be inside subfolders of images. This is required by the data loader.
    dataset = datasets.ImageFolder("./images")
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=params.lr_base*0.01, betas=(0.0, 0.99)
    )
    '''g_optimizer.add_param_group(
        {
            'params': generator.style.parameters(),
            'lr': params.lr_base * 0.01,
            'mult': 0.01,
        }
    )'''

    d_optimizer = optim.Adam(discriminator.parameters(), lr=params.lr_base, betas=(0.0, 0.99))

    g_running = StyledGenerator(params.code_size).to(device)
    g_running.train(False)
    accumulate(g_running, generator, 0)

    step = int(math.log2(params.init_size)) - 2
    resolution = 4 * 2 ** step

    adjust_lr(g_optimizer, params.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, params.lr.get(resolution, 0.001))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    rn = params.phase * 10

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0
    alpha = 0
    used_sample = 0

    image_idx = 0

    loader = sample_data(
        dataset, 16, resolution
    )
    data_loader = iter(loader)

    for i in range(rn):
        discriminator.zero_grad()
        alpha = min(1, 1 / params.phase * (used_sample + 1))

        if used_sample > params.phase * 2:
            step += 1

            if step > int(math.log2(params.max_size)) - 2:
                step = int(math.log2(params.max_size)) - 2

            else:
                alpha = 0
                used_sample = 0

            resolution = 4 * 2 ** step

            batch_size = params.batch_size[resolution]

            loader = sample_data(
                dataset, batch_size, resolution
            )
            data_loader = iter(loader)

            torch.save(
                {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                },
                f'{checkpoint_dir}/train_step-{step}.model',
            )

            adjust_lr(g_optimizer, params.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, params.lr.get(resolution, 0.001))

        try:
            real_image, label = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, label = next(data_loader)


        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.to(device)

        real_predict = discriminator(real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
        (-real_predict).backward()


        gen_in1, gen_in2 = torch.randn(2, b_size, params.code_size, device=device).chunk(
            2, 0
        )
        gen_in1 = gen_in1.squeeze(0)
        gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        fake_predict = fake_predict.mean()
        fake_predict.backward()

        eps = torch.rand(b_size, 1, 1, 1).to(device)
        x_hat = eps * real_image.data + (1 - eps) * fake_image.data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = torch.autograd.grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
        )[0]
        grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
        ).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val = grad_penalty.item()
        disc_loss_val = (real_predict - fake_predict).item()

        d_optimizer.step()

        if (i + 1) % params.n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)

            predict = discriminator(fake_image, step=step, alpha=alpha)

            loss = -predict.mean()

            gen_loss_val = loss.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 100 == 0:
            new_images = []

            gen_i, gen_j = params.gen_sample.get(resolution, (10, 5))

            with torch.no_grad():
                for _ in range(gen_i):
                    new_images.append(
                        g_running(
                            torch.randn(gen_j, params.code_size).to(device), step=step, alpha=alpha
                        ).data.cpu()
                    )

            utils.save_image(
                torch.cat(new_images, 0),
                f'{sample_dir}/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 10000 == 0:
            torch.save(
                g_running.state_dict(), f'checkpoint/{str(i + 1).zfill(6)}.model'
            )

        state_msg = (
            f'I: {i}, Size: {4 * 2 ** step}; Gen loss: {gen_loss_val:.3f}; Disc loss: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )
        print(state_msg)


def main():
    args = TrainParams()

    generator = StyledGenerator(args.code_size).to(device)
    discriminator = Discriminator().to(device)

    train(args, generator, discriminator)


if __name__ == '__main__':
    main()
