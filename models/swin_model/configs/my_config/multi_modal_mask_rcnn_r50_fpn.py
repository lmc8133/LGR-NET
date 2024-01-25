# model settings
backbone=dict(
    type='MuResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch')
    
# backbone=dict(
#     type='MuSwinTransformer',
#     embed_dim=96,
#     depths=[2, 2, 6, 2],
#     num_heads=[3, 6, 12, 24],
#     window_size=7,
#     mlp_ratio=4.,
#     qkv_bias=True,
#     qk_scale=None,
#     drop_rate=0.,
#     attn_drop_rate=0.,
#     drop_path_rate=0.2,
#     ape=False,  # if true, add absolute position embedding to patch embedding
#     patch_norm=True,
#     out_indices=(0, 1, 2, 3),
#     use_checkpoint=False)

neck=dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=4)