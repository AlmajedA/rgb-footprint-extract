# Ensure all pretrained weights are located in the weights/ directory

# Example for sample Urban3D image
CUDA_VISIBLE_DEVICES=0 python run_deeplab.py --inference --backbone=drn_c42 --out-stride=8 \
    --workers=2 --epochs=1 --test-batch-size=1 --gpu-ids=0 --resume=_crowdAI_deeplab_drn_c42_wce_dicef0.5 --best-miou \
    --window-size=300 --stride=300 \
    --input-filename='.\imgs\133.npy' \
    --output-filename='.\imgs\building83ss61.png' 
