import datetime
import os
import sys
import argparse
import logging

import cv2

import torch
import torch.utils.data
import torch.optim as optim

# from torchsummary import summary  # HybridGraspNet 返回字典，不兼容 torchsummary

import tensorboardX

from utils.visualisation.gridshow import gridshow

from utils.dataset_processing import evaluation
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='ggcnn', help='Network Name in .models')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=250, help='Validation Batches')

    # Model-specific parameters (for Parallel model)
    parser.add_argument('--fusion-type', type=str, default='simple',
                        choices=['simple', 'attention', 'cross_attention'],
                        help='Fusion type for parallel model (simple/attention/cross_attention)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads for cross_attention fusion')
    parser.add_argument('--swin-size', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'],
                        help='Swin Transformer model size')

    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
    parser.add_argument('--vis', action='store_true', help='Visualise the training process')

    args = parser.parse_args()
    return args


def validate(net, device, val_data, batches_per_epoch):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc)

                loss = lossd['loss']

                results['loss'] += loss.item()/ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item()/ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])

                s = evaluation.calculate_iou_match(q_out, ang_out,
                                                   val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                                   no_grasps=1,
                                                   grasp_width=w_out,
                                                   )

                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx < batches_per_epoch:
        for x, y, _, _, _ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            if batch_idx % 100 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()

    # Vis window
    if args.vis:
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    train_dataset = Dataset(args.dataset_path, start=0.0, end=args.split, ds_rotate=args.ds_rotate,
                            random_rotate=True, random_zoom=True,
                            include_depth=args.use_depth, include_rgb=args.use_rgb)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=True,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1*args.use_depth + 3*args.use_rgb
    ggcnn = get_network(args.network)

    # 根据网络类型传递参数
    if args.network == 'parallel':
        # Parallel 模型支持 fusion_type, num_heads, swin_size
        net = ggcnn(
            input_channels=input_channels,
            fusion_type=args.fusion_type,
            num_heads=args.num_heads,
            swin_size=args.swin_size,
            use_pretrained=True
        )
        logging.info(f'Parallel Model Config:')
        logging.info(f'  - Fusion Type: {args.fusion_type}')
        logging.info(f'  - Num Heads: {args.num_heads}')
        logging.info(f'  - Swin Size: {args.swin_size}')
    elif args.network == 'hybrid':
        # Serial 模型支持 swin_size
        net = ggcnn(
            input_channels=input_channels,
            swin_size=args.swin_size,
            use_pretrained=True
        )
        logging.info(f'Hybrid Model Config:')
        logging.info(f'  - Swin Size: {args.swin_size}')
    else:
        # 其他模型（如 ggcnn）
        net = ggcnn(input_channels=input_channels)
    
    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # 学习率调度器：余弦退火 (CosineAnnealingLR)
    # 从初始 lr 平滑降低到 0，有助于模型收敛到更好的局部最优
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,  # 周期长度 = 训练轮数
        eta_min=1e-6        # 最小学习率
    )
    logging.info('Done')

    # Print model architecture.
    # HybridGraspNet 返回字典，torchsummary 不兼容，直接显示参数量
    total_params = sum(p.numel() for p in net.parameters())
    logging.info(f'模型参数量: {total_params:,}')
    
    # 保存架构信息
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    f.write(f'Network: {args.network}\n')
    f.write(f'Total parameters: {total_params:,}\n')
    f.write(f'Input channels: {input_channels}\n')
    f.write(f'Input size: 300x300\n')
    
    # 保存模型特定配置
    if args.network == 'parallel':
        f.write(f'\n[Parallel Model Config]\n')
        f.write(f'Fusion Type: {args.fusion_type}\n')
        f.write(f'Num Heads: {args.num_heads}\n')
        f.write(f'Swin Size: {args.swin_size}\n')
    elif args.network == 'hybrid':
        f.write(f'\n[Hybrid Model Config]\n')
        f.write(f'Swin Size: {args.swin_size}\n')
    
    # 保存训练配置
    f.write(f'\n[Training Config]\n')
    f.write(f'Batch size: {args.batch_size}\n')
    f.write(f'Epochs: {args.epochs}\n')
    f.write(f'Learning rate: 0.001 (Adam)\n')
    f.write(f'LR Scheduler: CosineAnnealingLR\n')
    f.write(f'Dataset: {args.dataset}\n')
    f.write(f'Use Depth: {args.use_depth}\n')
    f.write(f'Use RGB: {args.use_rgb}\n')
    f.close()

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        tb.add_scalar('learning_rate', current_lr, epoch)
        logging.info('Learning Rate: {:.6f}'.format(current_lr))
        
        # Log learned weights (if using uncertainty loss)
        if 'learned_weight_p' in train_results['losses']:
            tb.add_scalar('uncertainty/weight_p', train_results['losses']['learned_weight_p'], epoch)
            tb.add_scalar('uncertainty/weight_cos', train_results['losses']['learned_weight_cos'], epoch)
            tb.add_scalar('uncertainty/weight_sin', train_results['losses']['learned_weight_sin'], epoch)
            tb.add_scalar('uncertainty/weight_width', train_results['losses']['learned_weight_width'], epoch)
            tb.add_scalar('uncertainty/sigma_p', train_results['losses']['uncertainty_p'], epoch)
            tb.add_scalar('uncertainty/sigma_cos', train_results['losses']['uncertainty_cos'], epoch)
            tb.add_scalar('uncertainty/sigma_sin', train_results['losses']['uncertainty_sin'], epoch)
            tb.add_scalar('uncertainty/sigma_width', train_results['losses']['uncertainty_width'], epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.val_batches)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))

        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_statedict.pt' % (epoch, iou)))
            best_iou = iou
        
        # 学习率调度器步进
        scheduler.step()


if __name__ == '__main__':
    run()
