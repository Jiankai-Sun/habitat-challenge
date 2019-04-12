#python -u evaluate_ppo_vis.py \
#    --model-path "data/checkpoints/ckpt.835.pth" \
#    --sim-gpu-id 0 \
#    --pth-gpu-id 0 \
#    --num-processes 1 \
#    --count-test-episodes 100 \
#    --task-config "tasks/pointnav_gibson.yaml" \

#python -u evaluate_ppo_vis_single_env.py \
#    --model-path "data/checkpoints_gibson/ckpt.1178.pth" \
#    --sim-gpu-id 0 \
#    --pth-gpu-id 0 \
#    --num-processes 1 \
#    --count-test-episodes 1000 \
#    --task-config "tasks/pointnav_gibson.yaml" \

python -u evaluate_ppo_vis_single_env.py \
    --model-path "../models/checkpoints_mp3d/ckpt.849.pth" \
    --pth-gpu-id 0 \
    --num-processes 1 \
    --count-test-episodes 495 \
    --task-config "tasks/pointnav_mp3d_rgbd.yaml" \
