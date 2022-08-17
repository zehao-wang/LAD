NODE_RANK=0
NUM_GPUS=1
DATA_ROOT=../datasets
warmup_steps=50000
outdir=../out/REVERIE/experiments/pretrain/frt_gd_phase2
rt_img_dir=../room_type_feats.h5

# Phase 2 init ckpt
phase_ckpt=../out/REVERIE/experiments/pretrain/frt_gd_phase1/ckpts/model_step_48000.pt

# train
PYTHONPATH="../":$PYTHONPATH /usr/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train.py --world_size ${NUM_GPUS} \
    --output_dir $outdir \
    --model_config config/reverie_obj_model_config.json \
    --config config/reverie_pretrain_rt_gd_nodist.json \
    --use_rt_task \
    --vlnbert cmt \
    --use_clip_feat \
    --rt_embed_dir $rt_img_dir \
    --use_fix_rt_emb \
    --start_from 2 \
    --init_ckpt $phase_ckpt \
    --gd_warmup_steps $warmup_steps
