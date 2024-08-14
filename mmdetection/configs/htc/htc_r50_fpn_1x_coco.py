_base_ = './htc-without-semantic_r50_fpn_1x_coco.py'
model = dict(
    data_preprocessor=dict(pad_seg=True),
    roi_head=dict(
        semantic_roi_extractor=None,
        semantic_head=None
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img='train/', seg=''),
        pipeline=train_pipeline))
