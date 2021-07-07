_base_ = ['../_base_/models/mask_rcnn_r50pf_fpn_sk.py',
          '../_base_/datasets/sk_instance.py',
          '../_base_/schedules/schedule_1x.py',
          '../_base_/default_runtime.py']

# modify pf_cfg
model = dict(
    backbone=dict(
        type='ResNetPf',
        depth=50,
        pf_cfg=[64, 32, 32, 128, 51, 51, 128, 256, 153, 153, 153, 153, 256, 512, 512, 512]),
    )

# ============================== schedules =====================================
# optimizer
optimizer = dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    warmup_iters=86,
    warmup_ratio=0.02,
    step=[6, 11])  # start from 0
total_epochs = 18

# ============================== runtime =======================================
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
load_from = "work_dirs/prune/1/pruned-B.pth"
