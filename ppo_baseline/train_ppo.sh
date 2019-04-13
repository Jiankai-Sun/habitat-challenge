directory="data/"
if [ -d "$directory" ];
then
	echo "$directory found."
else
	echo "$directory not found."
fi

## gibson
#python -u train_ppo.py \
#    --use-gae \
#    --sim-gpu-id 0 \
#    --pth-gpu-id 1 \
#    --lr 2.5e-4 \
#    --clip-param 0.1 \
#    --value-loss-coef 0.5 \
#    --num-processes 10 \
#    --num-steps 128 \
#    --num-mini-batch 10 \
#    --num-updates 100000 \
#    --use-linear-lr-decay \
#    --use-linear-clip-decay \
#    --entropy-coef 0.01 \
#    --tb-log-dir "data/tb_log/tb_gibson_log" \
#    --log-file "data/train_gibson.log" \
#    --log-interval 5 \
#    --checkpoint-folder "data/checkpoints_gibson" \
#    --checkpoint-interval 50 \
#    --task-config "tasks/pointnav_gibson.yaml" \

## mp3d
#python -u train_ppo.py \
#    --use-gae \
#    --sim-gpu-id 2 \
#    --pth-gpu-id 3 \
#    --lr 2.5e-4 \
#    --clip-param 0.1 \
#    --value-loss-coef 0.5 \
#    --num-processes 10 \
#    --num-steps 128 \
#    --num-mini-batch 10 \
#    --num-updates 100000 \
#    --use-linear-lr-decay \
#    --use-linear-clip-decay \
#    --entropy-coef 0.01 \
#    --tb-log-dir "data/tb_log/tb_mp3d_log" \
#    --log-file "data/train_mp3d.log" \
#    --log-interval 5 \
#    --checkpoint-folder "data/checkpoints_mp3d" \
#    --checkpoint-interval 50 \
#    --task-config "tasks/pointnav_mp3d.yaml" \

python -u train_ppo.py \
    --use-gae \
    --use-icm \
    --sim-gpu-id 2 \
    --pth-gpu-id 3 \
    --lr 2.5e-4 \
    --clip-param 0.1 \
    --value-loss-coef 0.5 \
    --num-processes 2 \
    --num-steps 128 \
    --num-mini-batch 2 \
    --num-updates 100000 \
    --use-linear-lr-decay \
    --use-linear-clip-decay \
    --entropy-coef 0.01 \
    --tb-log-dir "data/tb_log/tb_gibson_curiosity_log" \
    --log-file "data/train_gibson_curiosity.log" \
    --log-interval 5 \
    --checkpoint-folder "data/checkpoints_gibson_curiosity" \
    --checkpoint-interval 50 \
    --task-config "tasks/pointnav_gibson_rgbd.yaml" \
