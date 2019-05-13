#python -u evaluate_debug.py \
#    --sim-gpu-id 0 \
#    --pth-gpu-id 0 \
#    --num-processes 1 \
#    --count-test-episodes 1000 \
#    --outdir data/slam_result/out_dir_num_erosion_2_mapsize_6000_original_recovery \
#    --task-config "tasks/pointnav_gibson_rgbd.yaml" \

#python -u evaluate_debug.py \
#    --sim-gpu-id 2 \
#    --pth-gpu-id 3 \
#    --num-processes 1 \
#    --count-test-episodes 1000 \
#    --outdir data/slam_result/out_dir_dstar \
#    --task-config "tasks/pointnav_gibson_rgbd.yaml" \

# tmp
python -u evaluate_debug.py \
    --sim-gpu-id 0 \
    --pth-gpu-id 0 \
    --num-processes 1 \
    --count-test-episodes 1000 \
    --outdir data/slam_result/out_dir_dstar \
    --task-config "tasks/pointnav_gibson_rgbd.yaml" \

#python -u evaluate_debug_rgb.py \
#    --sim-gpu-id 0 \
#    --pth-gpu-id 0 \
#    --num-processes 1 \
#    --count-test-episodes 1000 \
#    --outdir data/slam_result/out_dir_baseline+rgb2depth \
#    --task-config "tasks/pointnav_gibson_rgb.yaml" \
