import habitat
import cv2
import numpy as np
import os.path as osp

def visualize_traj(env, traj_coors, output_dir):
    """
    env: interative environment;
    traj_coors: list of each points' coordinate on the trajectory (coordinates are in the format of (x, _, y));
    output_dir: output directory of the topdown freespace map
    """
    topdown_freespace_map = np.uint8(np.zeros([512, 512, 3]))
    pix_f = [255, 255, 255]

    minx = 100
    miny = 100
    maxx = -100
    maxy = -100

    for _ in range(10000):
        x, _, y = env.sim.sample_navigable_point()
        if x < minx:
            minx = x
        if y < miny:
            miny = y
        if x > maxx:
            maxx = x
        if y > maxy:
            maxy = y

    for _ in range(100000):
        x, _, y = env.sim.sample_navigable_point()
        gx = int((x - minx) / (maxx - minx) * 511)
        gy = int((y - miny) / (maxy - miny) * 511)
        topdown_freespace_map[gx, gy] = pix_f

    for i, coor in enumerate(traj_coors):
        x, _, y = coor
        gx = int((x - minx) / (maxx - minx) * 511)
        gy = int((y - miny) / (maxy - miny) * 511)
        if i == 0:
            print(gx, gy)
            cv2.circle(topdown_freespace_map, (gx, gy), 8, (0, 126, 255), -1)
        elif i == len(traj_coors) - 1:
            cv2.circle(topdown_freespace_map, (gx, gy), 8, (255, 126, 0), -1)
        else:
            cv2.circle(topdown_freespace_map, (gx, gy), 6, (0, 255, 0), -1)
    cv2.imwrite(osp.join(output_dir, 'example_topdown_freespace.png'), topdown_freespace_map)

if __name__ == '__main__':
    env = habitat.Env(
        config=habitat.get_config(config_file="tasks/pointnav_mp3d.yaml")
    )
    traj_coors = [
        (2.5, 0.7, 0.3),
        (2.5, 0.7, 0.55),
        (2.5, 0.7, 0.80),
        (2.5, 0.7, 1.05),
        (2.5, 0.7, 1.3),
        (2.5, 0.7, 1.55),
    ]

    visualize_traj(env, traj_coors, './visualize')