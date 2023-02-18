# The name of experiment
name=VLT5

output=snap/vqa/$name

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    ../src/vqa.py \
        --from_scratch 1 \
        --distributed --multiGPU \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 30 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 20 \
        --valid_batch_size 100 \