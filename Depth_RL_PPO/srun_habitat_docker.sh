#nvidia-docker run --storage-opt size=100G -v /mnt/lustrenew/sunjiankai/Program/habitat-baseline:/habitat-baseline -v /mnt/lustrenew/sunjiankai/Program/habitat-api/data/scene_datasets/gibson:/habitat-api/data/scene_datasets/gibson -v /mnt/lustrenew/sunjiankai/Program/habitat-api/data/scene_datasets/mp3d:/habitat-api/data/scene_datasets/mp3d -v /mnt/lustrenew/sunjiankai/Program/habitat-api/data/datasets/pointnav/gibson:/habitat-api/data/datasets/pointnav/gibson -v /mnt/lustrenew/sunjiankai/Program/habitat-api/data/datasets/pointnav/mp3d:/habitat-api/data/datasets/pointnav/mp3d -v /mnt/lustrenew/sunjiankai/Program/habitat-api/data/datasets/pointnav/gibson.yaml:/habitat-api/data/datasets/pointnav/gibson.yaml -v /mnt/lustrenew/sunjiankai/Program/habitat-api/data/datasets/pointnav/mp3d.yaml:/habitat-api/data/datasets/pointnav/mp3d.yaml -it jack/habitat:v0.01 /bin/bash
srun --partition=AD --gres=gpu:4 --job-name=docker nvidia-docker run --storage-opt size=100G -v /mnt/lustrenew/sunjiankai/Program/habitat-baseline:/habitat-baseline -v /mnt/lustre/sunjiankai/lustrenew/Program/habitat-api/data:/habitat-api/data -it jack/habitat:v0.03 /bin/bash
