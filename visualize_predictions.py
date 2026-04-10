import argparse
import logging
import re
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

from models import get_network
from models.common import post_process_output
from utils.data import get_dataset
from utils.dataset_processing.grasp import detect_grasps

matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize grasp predictions and save images with predicted grasp rectangles.'
    )
    parser.add_argument('--model-path', type=str, required=True, help='Path to a model file or state_dict file.')
    parser.add_argument('--dataset', type=str, required=True, choices=['cornell', 'jacquard'],
                        help='Dataset name.')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to dataset root.')
    parser.add_argument('--output-dir', type=str, default='output/prediction_visualizations',
                        help='Directory to save rendered prediction images.')
    parser.add_argument('--sample-indices', type=int, nargs='+', default=[0],
                        help='One or more dataset indices to visualize.')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Dataset split start fraction; defaults to validation/test split.')
    parser.add_argument('--end', type=float, default=1.0,
                        help='Dataset split end fraction.')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Dataset rotation fraction used during train/test splitting.')
    parser.add_argument('--use-depth', type=int, default=1, help='Use depth channel (1/0).')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB channels (1/0).')
    parser.add_argument('--n-grasps', type=int, default=1, help='Number of predicted grasps to draw.')
    parser.add_argument('--device', type=str, default='auto', help='cpu, cuda, cuda:0, or auto.')
    parser.add_argument('--network', type=str, default=None, choices=['hybrid', 'parallel'],
                        help='Override auto-detected network type.')
    parser.add_argument('--swin-size', type=str, default=None, choices=['tiny', 'small', 'base'],
                        help='Override auto-detected Swin size for hybrid model.')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Attention heads for the parallel model if manual override is needed.')
    parser.add_argument('--use-gaam', action='store_true',
                        help='Force-enable GAAM when loading a plain state_dict.')
    parser.add_argument('--use-cf-gaam', action='store_true',
                        help='Force-enable CF-GAAM when loading a plain state_dict.')
    parser.add_argument('--num-peaks', type=int, default=5,
                        help='Peak count for CF-GAAM models.')
    parser.add_argument('--show-gt', action='store_true',
                        help='Overlay ground-truth grasp rectangles in green.')
    parser.add_argument('--save-heatmaps', action='store_true',
                        help='Also save a 2x2 summary figure with RGB/depth/Q/angle maps.')
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg != 'auto':
        return torch.device(device_arg)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint, None

    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'net', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return None, checkpoint[key]

        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return None, checkpoint

    raise ValueError(
        'Unsupported checkpoint format. Please pass a full model file or a plain state_dict/checkpoint dict.'
    )


def infer_input_channels(state_dict):
    candidate_keys = [
        'cnn_backbone.stage1.0.weight',
        'backbone.stage1.0.weight',
        'swin_branch.patch_embed.proj.weight',
        'swin.patch_embed.proj.weight',
    ]
    for key in candidate_keys:
        if key in state_dict:
            return int(state_dict[key].shape[1])
    return 1


def infer_network_type(state_dict):
    keys = state_dict.keys()
    if any(k.startswith('fusion.') or k.startswith('cnn_backbone.') for k in keys):
        return 'parallel'
    if any(k.startswith('channel_adapter.') or k.startswith('backbone.') for k in keys):
        return 'hybrid'
    raise ValueError('Could not infer network type from state_dict. Please specify --network.')


def infer_swin_size(state_dict):
    patch_key = 'swin.patch_embed.proj.weight'
    if patch_key not in state_dict:
        return 'tiny'

    embed_dim = int(state_dict[patch_key].shape[0])
    if embed_dim == 128:
        return 'base'

    max_block_idx = -1
    block_pattern = re.compile(r'^swin\.layers\.2\.blocks\.(\d+)\.')
    for key in state_dict:
        match = block_pattern.match(key)
        if match:
            max_block_idx = max(max_block_idx, int(match.group(1)))

    return 'small' if max_block_idx >= 17 else 'tiny'


def infer_attention_flags(state_dict):
    keys = list(state_dict.keys())
    use_cf_gaam = any('.cf_gaam' in k for k in keys)
    use_gaam = any('.gaam' in k for k in keys) and not use_cf_gaam
    return use_gaam, use_cf_gaam


def build_model_from_state_dict(args, state_dict):
    network = args.network or infer_network_type(state_dict)
    input_channels = infer_input_channels(state_dict)
    inferred_use_gaam, inferred_use_cf_gaam = infer_attention_flags(state_dict)

    use_cf_gaam = args.use_cf_gaam or inferred_use_cf_gaam
    use_gaam = args.use_gaam or inferred_use_gaam

    model_cls = get_network(network)
    if network == 'parallel':
        model = model_cls(
            input_channels=input_channels,
            num_heads=args.num_heads,
            use_pretrained=False,
            use_gaam=use_gaam,
            use_cf_gaam=use_cf_gaam,
            num_peaks=args.num_peaks,
        )
    else:
        swin_size = args.swin_size or infer_swin_size(state_dict)
        model = model_cls(
            input_channels=input_channels,
            swin_size=swin_size,
            use_pretrained=False,
            use_gaam=use_gaam,
            use_cf_gaam=use_cf_gaam,
            num_peaks=args.num_peaks,
        )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logging.warning('Missing keys while loading state_dict: %d', len(missing_keys))
    if unexpected_keys:
        logging.warning('Unexpected keys while loading state_dict: %d', len(unexpected_keys))

    return model, {
        'network': network,
        'input_channels': input_channels,
        'use_gaam': use_gaam,
        'use_cf_gaam': use_cf_gaam,
        'swin_size': args.swin_size or (infer_swin_size(state_dict) if network == 'hybrid' else None),
        'num_heads': args.num_heads if network == 'parallel' else None,
    }


def tensor_to_rgb_image(dataset, sample_idx, rot, zoom):
    rgb = dataset.get_rgb(sample_idx, rot, zoom, normalise=False)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    return rgb


def tensor_to_depth_image(dataset, sample_idx, rot, zoom):
    return dataset.get_depth(sample_idx, rot, zoom)


def choose_canvas_image(dataset, sample_idx, rot, zoom, use_rgb):
    if use_rgb:
        return tensor_to_rgb_image(dataset, sample_idx, rot, zoom), 'rgb'
    return tensor_to_depth_image(dataset, sample_idx, rot, zoom), 'depth'


def summarize_prediction(predicted_grasps, gt_bbs):
    summary = {
        'pred_center': None,
        'gt_center': None,
        'best_iou': 0.0,
        'success': False,
        'inference_time_ms': None,
    }

    if gt_bbs is not None:
        summary['gt_center'] = tuple(int(v) for v in gt_bbs.center[::-1])

    if not predicted_grasps:
        return summary

    best_grasp = predicted_grasps[0]
    best_iou = best_grasp.max_iou(gt_bbs) if gt_bbs is not None else 0.0

    for grasp in predicted_grasps:
        if gt_bbs is not None:
            grasp_iou = grasp.max_iou(gt_bbs)
            if grasp_iou > best_iou:
                best_iou = grasp_iou
                best_grasp = grasp

    summary['pred_center'] = tuple(int(v) for v in best_grasp.center[::-1])
    summary['best_iou'] = float(best_iou)
    summary['success'] = best_iou > 0.25
    return summary


def render_prediction(dataset, sample_idx, rot, zoom, q_img, ang_img, width_img,
                      no_grasps, output_path, use_rgb, show_gt, inference_time_ms=None):
    image, image_kind = choose_canvas_image(dataset, sample_idx, rot, zoom, use_rgb)
    predicted_grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=no_grasps)
    gt_bbs = dataset.get_gtbb(sample_idx, rot, zoom) if show_gt else None
    summary = summarize_prediction(predicted_grasps, gt_bbs)
    summary['inference_time_ms'] = inference_time_ms

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    if image_kind == 'rgb':
        ax.imshow(image)
    else:
        ax.imshow(image, cmap='gray')

    for grasp in predicted_grasps:
        grasp.plot(ax, color='red')

    if gt_bbs is not None:
        gt_bbs.plot(ax, color='lime')

    title_parts = [f'Sample {sample_idx}', f'Predicted grasps: {len(predicted_grasps)}']
    if summary['inference_time_ms'] is not None:
        title_parts.append(f"Inference: {summary['inference_time_ms']:.2f} ms")
    if summary['pred_center'] is not None:
        title_parts.append(f"Pred center: {summary['pred_center']}")
    if summary['gt_center'] is not None:
        title_parts.append(f"GT center: {summary['gt_center']}")
        title_parts.append(f"Best IoU: {summary['best_iou']:.3f}")
        title_parts.append(f"Result: {'SUCCESS' if summary['success'] else 'FAIL'}")

    ax.set_title(' | '.join(title_parts))
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return predicted_grasps, summary


def render_heatmaps(dataset, sample_idx, rot, zoom, rgb_img, depth_img, q_img, ang_img,
                    width_img, no_grasps, output_path, show_gt=False, inference_time_ms=None):
    predicted_grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=no_grasps)
    gt_bbs = dataset.get_gtbb(sample_idx, rot, zoom) if show_gt else None
    summary = summarize_prediction(predicted_grasps, gt_bbs)
    summary['inference_time_ms'] = inference_time_ms

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for grasp in predicted_grasps:
        grasp.plot(ax, color='red')
    if gt_bbs is not None:
        gt_bbs.plot(ax, color='lime')
    ax.set_title('RGB + Prediction')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(depth_img, cmap='gray')
    for grasp in predicted_grasps:
        grasp.plot(ax, color='red')
    if gt_bbs is not None:
        gt_bbs.plot(ax, color='lime')
    ax.set_title('Depth + Prediction')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    q_plot = ax.imshow(q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q Map')
    ax.axis('off')
    plt.colorbar(q_plot, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(2, 2, 4)
    angle_plot = ax.imshow(ang_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle Map')
    ax.axis('off')
    plt.colorbar(angle_plot, ax=ax, fraction=0.046, pad=0.04)

    suptitle_parts = [f'Sample {sample_idx}']
    if summary['inference_time_ms'] is not None:
        suptitle_parts.append(f"Inference: {summary['inference_time_ms']:.2f} ms")
    if summary['pred_center'] is not None:
        suptitle_parts.append(f"Pred center: {summary['pred_center']}")
    if summary['gt_center'] is not None:
        suptitle_parts.append(f"GT center: {summary['gt_center']}")
        suptitle_parts.append(f"Best IoU: {summary['best_iou']:.3f}")
        suptitle_parts.append(f"Result: {'SUCCESS' if summary['success'] else 'FAIL'}")
    fig.suptitle(' | '.join(suptitle_parts))

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_or_none, state_dict = load_checkpoint(args.model_path, device)
    if model_or_none is not None:
        model = model_or_none.to(device)
        model_info = {'network': type(model).__name__}
    else:
        model, model_info = build_model_from_state_dict(args, state_dict)
        model = model.to(device)

    model.eval()
    logging.info('Using device: %s', device)
    logging.info('Model info: %s', model_info)

    dataset_cls = get_dataset(args.dataset)
    dataset = dataset_cls(
        args.dataset_path,
        start=args.split,
        end=args.end,
        ds_rotate=args.ds_rotate,
        random_rotate=False,
        random_zoom=False,
        include_depth=bool(args.use_depth),
        include_rgb=bool(args.use_rgb),
    )
    logging.info('Dataset size in selected split: %d', len(dataset))

    with torch.no_grad():
        for sample_idx in args.sample_indices:
            if sample_idx < 0 or sample_idx >= len(dataset):
                raise IndexError(f'Sample index {sample_idx} is out of range for selected split size {len(dataset)}.')

            x, _, didx, rot, zoom = dataset[sample_idx]
            xc = x.unsqueeze(0).to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            pred = model(xc)
            # Some models (e.g. hybrid) predict at 224x224 while Cornell samples are 300x300.
            # Resize predictions back to the input image size before peak detection and plotting.
            target_size = xc.shape[2:]
            if pred['pos'].shape[2:] != target_size:
                pred = {
                    key: F.interpolate(value, size=target_size, mode='bilinear', align_corners=True)
                    for key, value in pred.items()
                }

            q_img, ang_img, width_img = post_process_output(
                pred['pos'], pred['cos'], pred['sin'], pred['width']
            )
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_time_ms = (time.perf_counter() - start_time) * 1000.0

            base_name = f'sample_{sample_idx:04d}'
            overlay_path = output_dir / f'{base_name}_prediction.png'
            grasps, summary = render_prediction(
                dataset=dataset,
                sample_idx=didx,
                rot=rot,
                zoom=zoom,
                q_img=q_img,
                ang_img=ang_img,
                width_img=width_img,
                no_grasps=args.n_grasps,
                output_path=overlay_path,
                use_rgb=bool(args.use_rgb),
                show_gt=args.show_gt,
                inference_time_ms=inference_time_ms,
            )

            logging.info('Saved prediction image: %s', overlay_path)
            logging.info('  inference_time_ms: %.2f', inference_time_ms)
            if summary['gt_center'] is not None:
                logging.info(
                    '  summary: pred_center=%s gt_center=%s best_iou=%.3f result=%s',
                    summary['pred_center'],
                    summary['gt_center'],
                    summary['best_iou'],
                    'SUCCESS' if summary['success'] else 'FAIL',
                )
            for grasp_idx, grasp_obj in enumerate(grasps):
                logging.info(
                    '  grasp[%d]: center=(%.1f, %.1f), angle=%.2f deg, length=%.1f, width=%.1f',
                    grasp_idx,
                    grasp_obj.center[1],
                    grasp_obj.center[0],
                    -grasp_obj.angle * 180.0 / np.pi,
                    grasp_obj.length,
                    grasp_obj.width,
                )

            if args.save_heatmaps:
                rgb_img = tensor_to_rgb_image(dataset, didx, rot, zoom)
                depth_img = tensor_to_depth_image(dataset, didx, rot, zoom)
                heatmap_path = output_dir / f'{base_name}_summary.png'
                render_heatmaps(
                    dataset=dataset,
                    sample_idx=didx,
                    rot=rot,
                    zoom=zoom,
                    rgb_img=rgb_img,
                    depth_img=depth_img,
                    q_img=q_img,
                    ang_img=ang_img,
                    width_img=width_img,
                    no_grasps=args.n_grasps,
                    output_path=heatmap_path,
                    show_gt=args.show_gt,
                    inference_time_ms=inference_time_ms,
                )
                logging.info('Saved summary image: %s', heatmap_path)


if __name__ == '__main__':
    main()
