_base_ = ['../_base_/models/mask_rcnn_r50pf_fpn_sk.py',  # model
          '../_base_/datasets/sk_instance.py']  # dataset

# ============================== schedules =====================================

# optimizer
optimizer = dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=86,
    warmup_ratio=0.02,
    step=[6, 11])  # start from 0
total_epochs = 12

# ============================== runtime =======================================

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

