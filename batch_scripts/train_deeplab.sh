# Example for Urban3D
CUDA_VISIBLE_DEVICES=0 python run_deeplab.py --backbone=drn_c42 --out-stride=8 --dataset=crowdAI \
    --workers=4 --loss-type=wce_dice --fbeta=0.1 --epochs=20 --batch-size=4 --test-batch-size=4 --weight-decay=1e-4 \
    --gpu-ids=0 --lr=1e-3 --loss-weights 1.0 1.0 --dropout 0.3 0.5 --incl-bounds \
    --checkname=paper_model_20 --data-root=./data/