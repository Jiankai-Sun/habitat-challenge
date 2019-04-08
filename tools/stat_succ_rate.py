import os

dataset = 'mp3d'  # 'gibson'
top_down_vis_dir = '/home/SENSETIME/sunjiankai/data/Program/habitat-api/baselines/data/{0}_result/top_down_vis'.format(dataset)

file_list = os.listdir(top_down_vis_dir)
counter = 0
success_counter = 0
for file_name in file_list:
    counter += 1
    success = file_name.split('.')[0].split('_')[1]
    # print(success)
    if success == 'False':
        success_counter += 1
    if counter % 44 == 0:
        print(success_counter)
        success_counter = 0

