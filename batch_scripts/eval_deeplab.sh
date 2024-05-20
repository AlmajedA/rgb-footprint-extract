# Ensure all pretrained weights are located in the weights/ directory

# Example for CrowdAI
CUDA_VISIBLE_DEVICES=0 python run_deeplab.py --evaluate --backbone=drn_c42 --out-stride=8 \
    --workers=2 --epochs=1 --test-batch-size=3 --gpu-ids=0 \
    --checkname=paper_model_20 --dataset=crowdAI --resume=paper_model_20
