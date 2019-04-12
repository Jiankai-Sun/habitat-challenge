python -u evaluate_ppo.py \
    --model-path "../models/checkpoints_gibson/ckpt.1178.pth" \
    --pth-gpu-id 0 \
    --num-processes 1 \
    --count-test-episodes 1000 \
    --task-config "tasks/pointnav_gibson_rgbd.yaml" \


#python -u evaluate_ppo.py \
#    --model-path "../models/checkpoints_mp3d/ckpt.849.pth" \
#    --sim-gpu-id 0 \
#    --pth-gpu-id 0 \
#    --num-processes 1 \
#    --count-test-episodes 495 \
#    --task-config "tasks/pointnav_mp3d_rgbd.yaml" \
