"""Reconstruct one experimental event/image pair without a Lightning trainer."""

import argparse
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
import torch
from torchvision import transforms

from models.resunet import ResUnet
from models.unet_attention import Unet
import np_transforms
import utils


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a network state dict or a Lightning checkpoint')
    if 'state_dict' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        state = checkpoint['state_dict']
        if not all(key.startswith('model.') for key in state):
            raise ValueError('Expected a Lightning checkpoint containing only model.* weights')
        state = {key[len('model.'):]: value for key, value in state.items()}
    else:
        state = checkpoint
        if 'init_conv.weight' in state:
            input_weight, vanilla_unet = state['init_conv.weight'], False
        elif 'input_layer.0.weight' in state:
            input_weight, vanilla_unet = state['input_layer.0.weight'], True
        else:
            raise ValueError('State dict does not contain a supported network input convolution')
        if not isinstance(input_weight, torch.Tensor) or input_weight.ndim != 4:
            raise ValueError('Expected a four-dimensional input convolution weight')
        hparams = {'dim': input_weight.shape[0], 'in_channels': input_weight.shape[1],
                   'vanilla_unet': vanilla_unet}
    if hparams.get('vanilla_unet', False):
        model = ResUnet(in_channels=hparams['in_channels'], out_channels=1,
                        dim=hparams['dim'], conv_kernel_size=3)
    else:
        # The upstream flash helper disables the float32 attention kernel on A100.
        model = Unet(dim=hparams['dim'], dim_mults=(1, 2, 4, 8),
                     in_channels=hparams['in_channels'], out_channels=1,
                     flash_attn=False)
    model.load_state_dict(state, strict=True)
    metadata = {
        'hyper_parameters': hparams,
        'strict_load': True,
        'state_tensors': len(state),
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
    }
    if 'state_dict' in checkpoint:
        metadata.update(epoch=checkpoint.get('epoch'), global_step=checkpoint.get('global_step'))
    return model.eval().to(device), metadata


def validate_settings(args, in_channels):
    if args.time_bin < 1:
        raise ValueError('--time-bin must be positive')
    channels = args.time_bin * (2 if args.polarity else 1) + int(args.time_std)
    if channels != in_channels:
        raise ValueError(f'Preprocessing produces {channels} channels; checkpoint expects {in_channels}')
    if (args.pixel_bin < 1 or 720 % args.pixel_bin or 1280 % args.pixel_bin
            or min(720 // args.pixel_bin, 1280 // args.pixel_bin) < 8):
        raise ValueError('--pixel-bin must divide both 720 and 1280 and leave at least 8 pixels per axis')
    if (not math.isfinite(args.integration_time_s) or not math.isfinite(args.start_time_s)
            or args.integration_time_s <= 0 or args.start_time_s < 0
            or args.start_time_s + args.integration_time_s > 10):
        raise ValueError('The requested positive-duration window must be inside the 10-second recording')
    return (720 // args.pixel_bin // 8 * 8, 1280 // args.pixel_bin // 8 * 8)


def prepare_sample(args, in_channels):
    shape = validate_settings(args, in_channels)
    for path in (args.events, args.image):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.events.stem != args.image.stem or args.events.suffix != '.h5':
        raise ValueError('--events and --image must be a same-stem .h5/image pair')
    with h5py.File(args.events, 'r') as recording:
        attrs = recording['event_buffers'].attrs
        if (attrs['height'], attrs['width']) != (720, 1280):
            raise ValueError('The experimental preprocessing requires a 720x1280 event sensor')
    dataset = utils.EventImagePairDataset(
        image_folder=str(args.image.parent), event_folder=str(args.events.parent),
        image_names=[args.image.name], img_suffix=args.image.suffix,
        integration_time_s=args.integration_time_s, start_time_s=args.start_time_s,
        pixel_bin=args.pixel_bin, time_bin=args.time_bin, polarity=args.polarity,
        std_channel=args.time_std,
        transform=transforms.Compose([
            np_transforms.CenterCrop(shape),
            utils.EventCountNormalization(),
        ]),
    )
    if len(dataset) != 1:
        raise ValueError('The reference image must have the dataset resolution, 1080x1920')
    counts, truth, duration = dataset[0]
    if not torch.isfinite(counts).all() or not torch.isfinite(truth).all():
        raise ValueError('Preprocessing produced non-finite inputs or reference pixels')
    return counts.unsqueeze(0), truth[0].numpy(), duration.unsqueeze(0)


def to_image(array):
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


def run(args):
    device = torch.device(args.device)
    model, metadata = load_model(args.checkpoint, device)
    counts, truth, duration = prepare_sample(args, metadata['hyper_parameters']['in_channels'])
    counts_device, duration_device = counts.to(device), duration.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model(counts_device, duration_device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    expected_shape = (1, 1, *truth.shape)
    if tuple(prediction.shape) != expected_shape or not torch.isfinite(prediction).all():
        raise RuntimeError(f'Expected finite output of shape {expected_shape}, got {tuple(prediction.shape)}')
    reconstruction = prediction[0, 0].cpu().numpy()
    mse = float(np.mean((reconstruction.astype(np.float64) - truth) ** 2))
    noise = counts[0, :args.time_bin * (2 if args.polarity else 1)].sum(dim=0).numpy()
    noise = np.log1p(noise)
    noise /= max(1.0, float(noise.max()))
    panels = [to_image(noise), to_image(reconstruction), to_image(truth)]
    width, height = panels[0].size
    comparison = Image.new('RGB', (width * 3, height + 24), 'white')
    draw = ImageDraw.Draw(comparison)
    for index, (panel, label) in enumerate(zip(panels, ['Events (log-scaled)', 'Reconstruction', 'Reference'])):
        comparison.paste(panel, (index * width, 24))
        draw.text((index * width + 8, 5), label, fill='black')
    report = {
        'checkpoint': str(args.checkpoint.resolve()),
        'events': str(args.events.resolve()),
        'reference': str(args.image.resolve()),
        'checkpoint_metadata': metadata,
        'preprocessing': {
            'pixel_bin': args.pixel_bin, 'time_bin': args.time_bin,
            'polarity': args.polarity, 'time_std': args.time_std,
            'start_time_s': args.start_time_s, 'integration_time_s': args.integration_time_s,
            'normalization': 'repository EventCountNormalization() (fixed training scale)',
        },
        'device': str(device), 'torch_version': torch.__version__,
        'input_shape': list(counts.shape), 'output_shape': list(prediction.shape),
        'forward_seconds': elapsed,
        'prediction_min': float(reconstruction.min()),
        'prediction_max': float(reconstruction.max()),
        'metrics_unclipped': {
            'mse': mse, 'psnr_db': -10 * math.log10(mse) if mse else 'inf',
            'ssim': float(structural_similarity(truth, reconstruction, data_range=1.0)),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'reconstruction.npy', reconstruction)
    for panel, name in zip(panels, ['events.png', 'reconstruction.png', 'reference.png']):
        panel.save(args.output_dir / name)
    comparison.save(args.output_dir / 'comparison.png')
    text = json.dumps(report, indent=2, allow_nan=False)
    (args.output_dir / 'report.json').write_text(text + '\n')
    print(text)
    print(f'Outputs saved to {args.output_dir.resolve()}')
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Plain network state dict or original Lightning checkpoint')
    parser.add_argument('--events', type=Path, required=True, help='One experimental .h5 recording')
    parser.add_argument('--image', type=Path, required=True, help='Its paired reference image (for preprocessing/evaluation)')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/example'))
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pixel-bin', type=int, default=2)
    parser.add_argument('--time-bin', type=int, default=1)
    parser.add_argument('--polarity', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--time-std', action='store_true')
    parser.add_argument('--start-time-s', type=float, default=5.0)
    parser.add_argument('--integration-time-s', type=float, default=1.0)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
