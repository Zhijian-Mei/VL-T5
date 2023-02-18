# The name of experiment
name=VLT5

output=snap/vcr/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/vcr.py \
        --from_scratch 1 \
        --distributed --multiGPU --fp16 \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --batch_size 4 \
        --valid_batch_size 20 \
        --max_text_length 100 \