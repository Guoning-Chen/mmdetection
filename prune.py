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
        self.config = 'configs/mask_rcnn/r50_fpn.py'
        self.checkpoint = 'work_dirs/r50_fpn/epoch_12.pth'
        self.backbone = 'ResNet50'
        self.prs = '00-00-90-90'
        self.result_path = 'work_dirs/r50_fpn/r50_prs0099.pth'


def get_mask(m: nn.Conv2d, last_mask, num_keep):
    # 获取经过上一层剪枝后，本层剩余的权重
    idx = np.squeeze(np.argwhere(np.asarray(last_mask.cpu().numpy()))).tolist()
    weight_copy = m.weight.data[:, idx, :, :].abs().clone().cpu().numpy()

    # 计算本层需要保留的通道
    l1_norm = np.sum(weight_copy, axis=(1, 2, 3))  # out * 1
    idx_increase = np.argsort(l1_norm)  # 将所有 channel按 l1_norm升序排列后的索引
    idx_reserved = idx_increase[::-1][:num_keep]  # 取降序的前 num_keep个索引

    # 生成 mask
    out_channels = m.weight.data.shape[0]
    mask = torch.zeros(out_channels)
    mask[idx_reserved.tolist()] = 1  # 将待保留的 channel置 1

    return mask


def str2list(prs_str):
    assert len(prs_str) == 11, "plan format: xx-xx-xx-xx!"
    splits = prs_str.split('-')
    return [float(s)/100 for s in splits]


def prune_top2_layers(arch, net, skip_block, prs, cuda=True):
    """
    Args:
        arch: (str) must be supported.
        net: (nn.Module).
        skip_block: (list of int) ids of blocks to skip.
        prs: (list of float) pruning ratios of all stages.
        num_class: (int).
        cuda: (bool).

    Returns:
        (list of int) cfg.
        (nn.Module) pruned model.
    """
    assert arch in ['ResNet50', 'ResNet101'], 'Wrong arch!'

    layer_id = 1
    cfg = []  # list of int, 长度等于 block数量
    cfg_mask = []  # 长度 = block数量，元素为长度等于 channel数的一维 Tensor

    # 由于 net.named_modules()是按顺序遍历，因此 downsample跟在 third layer之后。
    # downsample的 layer_id 就是 third layer的 id
    # downsample_layers = [4, 13, 25, 43]
    skip = []
    for block_id in skip_block:
        skip += [2 + 3 * block_id, 3 + 3 * block_id, 4 + 3 * block_id]

    for name, m in net.named_modules():
        # 只对 Conv2d层操作
        if not isinstance(m, nn.Conv2d):
            continue
        elif isinstance(m, nn.Conv2d) & ('downsample' in name):
            continue

        out_channels = m.weight.data.shape[0]

        if layer_id == 1:  # layer 1
            cfg_mask.append(torch.ones(out_channels))
            cfg.append(out_channels)
        elif (layer_id in skip) | (layer_id % 3 == 1):  # 需要跳过的层
            cfg_mask.append(torch.ones(out_channels))
            cfg.append(out_channels)
        else:
            stage = -1
            layer_borders = [10, 22, 40, 49]  # 每个 stage的最右一个 layer
            for border in layer_borders:
                if layer_id <= border:
                    stage = layer_borders.index(border)
                    break
            pr = prs[stage]
            num_keep = int(out_channels * (1 - pr))
            mask = get_mask(m=m, last_mask=cfg_mask[-1], num_keep=num_keep)
            cfg_mask.append(mask)
            cfg.append(num_keep)
        layer_id += 1

    state = {'ResNet50': 50, 'ResNet101': 101}
    pruned_net = ResNetPf(depth=state[arch], pf_cfg=cfg[1:])

    if cuda:
        pruned_net.cuda()

    layer_id = 1
    for (name0, m0), (name1, m1) in zip(net.named_modules(),
                                        pruned_net.named_modules()):
        assert name0 == name1, 'not the same modules!'

        if not isinstance(m0, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            continue
        if isinstance(m0, nn.Conv2d):
            if 'downsample' in name0:
                m1.weight.data = m0.weight.data.clone()
                continue
            mask = cfg_mask[layer_id - 1]
            if layer_id == 1:
                last_mask = torch.ones(3)
            else:
                last_mask = cfg_mask[layer_id - 2]
            idx = np.squeeze(np.argwhere(np.asarray(mask))).tolist()
            last_idx = np.squeeze(np.argwhere(np.asarray(last_mask))).tolist()
            w = m0.weight.data[:, last_idx, :, :].clone()
            m1.weight.data = w[idx, :, :, :].clone()
        elif isinstance(m0, nn.BatchNorm2d):
            if 'downsample' in name0:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
                continue
            mask = cfg_mask[layer_id - 1]
            idx = np.squeeze(np.argwhere(np.asarray(mask))).tolist()
            m1.weight.data = m0.weight.data[idx].clone()
            m1.bias.data = m0.bias.data[idx].clone()
            m1.running_mean = m0.running_mean[idx].clone()
            m1.running_var = m0.running_var[idx].clone()
            layer_id += 1
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

    return cfg, pruned_net


def prune_mask_rcnn_only(args: PruneParams):
    """
    Just prune without retraining.
    Args:
        args: (PruneParams).

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
    print('Before pruning, Backbone Params = %.2fM' % (num_before/1E6))

    # PRUNE FILTERS
    # func = {"ResNet50": prune_resnet50, "ResNet101": prune_resnet101}
    assert args.backbone in ['ResNet50', 'ResNet101'], "Wrong backbone type!"
    skip = {'ResNet34': [2, 8, 14, 16, 26, 28, 30, 32],
            'ResNet50': [2, 11, 20, 23, 89, 92, 95, 98],
            'ResNet101': [2, 11, 20, 23, 89, 92, 95, 98]}
    pf_cfg, new_backbone = prune_top2_layers(
        arch=args.backbone, net=model.backbone, skip_block=skip[args.backbone],
        prs=str2list(args.prs), cuda=True)
    model.backbone = new_backbone

    num_after = sum([p.nelement() for p in model.backbone.parameters()])
    print('After  pruning: Backbone Params = %.2fM' % (num_after/1E6))
    print("Prune rate: %.2f%%" % ((num_before-num_after)/num_before*100))

    # replace checkpoint['state_dict']
    checkpoint['state_dict'] = cp.weights_to_cpu(cp.get_state_dict(model))
    mmcv.mkdir_or_exist(osp.dirname(args.result_path))

    # save and immediately flush buffer
    torch.save(checkpoint, args.result_path)
    with open(args.result_path.split('.')[0] + '_cfg.txt', 'w') as f:
        f.write(str(pf_cfg))


def prune_mask_rcnn_with_retrain(args: PruneParams, steps):
    pass


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
    # # prune_mask_rcnn_only(args=PruneParams())
    # count_params(pth_path='work_dirs/r50_fpn/epoch_12.pth')
    # # count_params(pth_path='work_dirs/r50pf_fpn/ep36_5e-3/epoch_36.pth')
    # # print_backbone('work_dirs/r101_fpn/r101-pruned-B.pth')
    count_time()