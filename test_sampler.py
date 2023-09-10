import functools
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


# Modified from diffusers.utils.randn_tensor
def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def randn_tensor_like(x: torch.Tensor, generator: Optional[torch.Generator] = None):
    shape = x.shape
    generator = generator
    device = x.device
    dtype=x.dtype
    return randn_tensor(shape, generator, device, dtype)


def dummy_model():
    def model(sample, t, *args, **kwargs):
        # if t is a tensor, match the number of dimensions of sample
        if isinstance(t, torch.Tensor):
            num_dims = len(sample.shape)
            # pad t with 1s to match num_dims
            t = t.reshape(-1, *(1,) * (num_dims - 1)).to(sample.device).to(sample.dtype)

        return sample * t / (t + 1)

    return model


def dummy_sample_deter(batch_size=4, num_channels=3, height=8, width=8):
    num_elems = batch_size * num_channels * height * width
    sample = torch.arange(num_elems)
    sample = sample.reshape(num_channels, height, width, batch_size)
    sample = sample / num_elems
    sample = sample.permute(3, 0, 1, 2)

    return sample


class DummyModel:
    def __init__(
        self,
        img_resolution=8,
        img_channels=3,
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
    ):
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = dummy_model()

    def __call__(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        return self.model(x, sigma)
    
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def main(args):
    model = DummyModel(
        img_resolution=args.img_res,
        img_channels=args.num_channels,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data,
    )
    sample = dummy_sample_deter(args.batch_size, args.num_channels, args.img_res, args.img_res)

    generator = torch.manual_seed(args.seed)
    randn_like_fn = functools.partial(randn_tensor_like, generator=generator)

    images = edm_sampler(
        net=model,
        latents=sample,
        class_labels=None,
        randn_like=randn_like_fn,
        num_steps=args.num_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )

    expected_sum = torch.sum(torch.abs(images))
    expected_mean = torch.mean(torch.abs(images))

    print(f"Expected sum: {expected_sum}")
    print(f"Expected mean: {expected_mean}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_res", type=int, default=8)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()


    main(args)