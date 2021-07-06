import torch
import torch.nn as nn

import numpy as np
import os.path as osp

import mmcv
from mmdet.models import build_detector
from mmdet.models.backbones import ResNetPf
import mmcv.runner.checkpoint as cp

# used by count_time
from tools.test import parse_args
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset)
from mmdet.models import build_detector

class PruneParams:
    def __init__(self):
        self.config = 'configs/mask_rcnn/mask_rcnn_r50pf_fpn_1x_sk.py'
        self.checkpoint = 'work_dirs/r50pf_fpn_1x_sk/epoch_12.pth'
        self.plan = 'B'
        self.checkpoint_path = 'work_dirs/r50pf_fpn_1x_sk/pruned-B.pth'


def prune_resnet50(net, plan):
    """
    Args:
        net: (nn.Module) ResNet50 network to be pruned.
        plan: (str) 'A' or 'B'.
        num_class: (int).

    Returns:
        (nn.Module) pruned model
        (list of int) cfg
    """
    assert plan in ['A', 'B'], 'plan wrong!'
    skip = {
        'A': [2, 11, 20, 23, 38, 41, 44, 47],
        'B': [2, 11, 20, 23, 38, 41, 44, 47],
    }
    prune_prob = {
        'A': [0.3, 0.3, 0.3, 0.],
        'B': [0.5, 0.6, 0.4, 0.],
    }

    layer_id = 1
    cfg = []
    cfg_mask = []
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) & ('downsample' not in name):
            out_channels = m.weight.data.shape[0]
            if layer_id in skip[plan]:
                cfg_mask.append(torch.ones(out_channels))
                cfg.append(out_channels)
                layer_id += 1
                continue
            if layer_id % 3 == 2:  # first layer of each block
                if layer_id <= 10:
                    stage = 0
                elif layer_id <= 22:
                    stage = 1
                elif layer_id <= 40:
                    stage = 2
                else:
                    stage = 3
                prune_prob_stage = prune_prob[plan][stage]
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                num_keep = int(out_channels * (1 - prune_prob_stage))
                arg_max = np.argsort(L1_norm)
                arg_max_rev = arg_max[::-1][:num_keep]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                cfg.append(num_keep)
                layer_id += 1
                continue
            layer_id += 1

    new_net = ResNetPf(depth=50, pf_cfg=cfg)

    block_id = 0  # 0~len(cfg)
    layer_id = 1  # 1~50
    for (name0, m0), (name1, m1) in zip(net.named_modules(),
                                        new_net.named_modules()):
        assert name0 == name1, 'not the same modules!'
        if isinstance(m0, nn.Conv2d) & ('downsample' not in name0):
            if layer_id == 1:  # layer 1
                m1.weight.data = m0.weight.data.clone()
                layer_id += 1
            elif layer_id % 3 == 2:  # the first layer of bottleneck block
                mask = cfg_mask[block_id]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[idx.tolist(), :, :, :].clone()
                m1.weight.data = w.clone()
                block_id += 1
                layer_id += 1
            elif layer_id % 3 == 0:  # the second layer of bottleneck block
                mask = cfg_mask[block_id - 1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[:, idx.tolist(), :, :].clone()
                m1.weight.data = w.clone()
                layer_id += 1
            elif layer_id % 3 == 1:  # the third layer of bottleneck block
                m1.weight.data = m0.weight.data.clone()
                layer_id += 1
        elif isinstance(m0, nn.BatchNorm2d):
            if layer_id % 3 == 0:  # BatchNorm2d behind the first layer
                mask = cfg_mask[block_id - 1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.weight.data = m0.weight.data[idx.tolist()].clone()
                m1.bias.data = m0.bias.data[idx.tolist()].clone()
                m1.running_mean = m0.running_mean[idx.tolist()].clone()
                m1.running_var = m0.running_var[idx.tolist()].clone()
            else:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
    return cfg, new_net


def prune_mask_rcnn(args: PruneParams):
    """
    Just prune without retraining.
    Args:
        args: (PruneResnetParams).

    Returns: (MaskRCNN) pruned model in cuda.
    """
    cfg = mmcv.Config.fromfile(args.config)
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    assert cfg.model['type'] == 'MaskRCNN', 'model type should be MaskRCNN!'

    # load checkpoint
    checkpoint = cp.load_checkpoint(model=model, filename=args.checkpoint)

    num_before = sum([p.nelement() for p in model.backbone.parameters()])
    print('Before pruning: Params num = ', num_before)

    # PRUNE FILTERS
    pf_cfg, new_resnet50 = prune_resnet50(net=model.backbone, plan=args.plan)
    model.backbone = new_resnet50

    num_after = sum([p.nelement() for p in model.backbone.parameters()])
    print('After  pruning: Params num = ', num_after)
    print("Prune rate: ", (num_before - num_after) / num_before)

    # replace checkpoint['state_dict']
    checkpoint['state_dict'] = cp.weights_to_cpu(cp.get_state_dict(model))
    mmcv.mkdir_or_exist(osp.dirname(args.checkpoint_path))

    # save and immediately flush buffer
    torch.save(checkpoint, args.checkpoint_path)
    with open(args.checkpoint_path.split('.')[0] + '-pf.txt', 'w') as f:
        f.write(str(pf_cfg))


def print_backbone(pth_path):
    print("\n%s\n" % pth_path)
    state = torch.load(pth_path)['state_dict']
    for key, value in state.items():
        if 'backbone' not in key:
            break
        print(key, value.shape)


def count_params(pth_path):
    state = torch.load(pth_path)
    params = {'backbone': 0,
              'neck': 0,
              'rpn_head': 0,
              'roi_head': 0}
    for key, value in state['state_dict'].items():
        num_params = 1
        for x in value.shape:
            num_params = num_params * x
        for part_name in params.keys():
            if part_name in key:
                params[part_name] += num_params
                break

    total = 0
    for key, value in params.items():
        total += value
        print(key, ': ', value / 1E6, 'M')
    print('all: ', total / 1E6, 'M')


def count_time():
    # //========================copy for tools/test.py========================//
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # //===========================end copy===================================//
    distributed = False
    assert args.launcher == 'none', "launcher in test.py must be 'none'"

    # build the dataloader
    samples_per_gpu = 1
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    assert fp16_cfg is not None
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES


    model = MMDataParallel(model, device_ids=[0])

    model.eval()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)


if __name__ == '__main__':
    # prune_mask_rcnn(args=PruneParams())
    count_params(pth_path='work_dirs/r50pf_fpn_1x_sk/pruned-B.pth')