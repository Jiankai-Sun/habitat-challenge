# import cv2, ctypes, logging, os, pickle
# from numpy import ma
# from collections import OrderedDict
# import scipy, skfmm
# import matplotlib.pyplot as plt
# from a_star import AStar, Point
from a_star_lib import pyastar
from skimage.morphology import binary_closing, disk
import numpy as np

step_size = 5
num_rots = 36

def subplot(plt, Y_X, sz_y_sz_x=(10, 10)):
    Y, X = Y_X
    sz_y, sz_x = sz_y_sz_x
    plt.rcParams['figure.figsize'] = (X * sz_x, Y * sz_y)
    fig, axes = plt.subplots(Y, X)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig, axes


class ASTAR_Planner():
    def __init__(self, traversible, num_rots, start, goal):
        self.traversible = traversible
        self.angle_value = [0, 2.0 * np.pi / num_rots, -2.0 * np.pi / num_rots, 0]
        self.du = step_size
        self.num_rots = num_rots
        # self.action_list = self.search_actions()
        self.current_loc = start  # (int(start[0]), int(start[1]))
        self.goal_loc = goal  # (int(goal[0]), int(goal[1]))

    def set_goal(self, goal):
        # traversible_ma = ma.masked_values(self.traversible*1, 0)
        # goal_x, goal_y = int(goal[0]),int(goal[1])
        # traversible_ma[goal_y, goal_x] = 0
        # dd = skfmm.distance(traversible_ma, dx=1)
        # dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        # dd = ma.filled(dd, np.max(dd)+1)
        # self.fmm_dist = dd
        # self.goal = goal
        # return dd_mask
        pass

    def get_action(self, step_num):
        grid = self.traversible.astype(np.float32)
        grid_tmp = grid
        # print(grid_tmp.min(), grid_tmp.max())

        current_loc = (self.current_loc[:2]).astype(np.int64)
        goal_loc = self.goal_loc.astype(np.int64)
        # current_loc = current_loc[1, 0]
        # goal_loc = goal_loc[1, 0]
        # print('current_loc: ', current_loc, 'goal_loc: ', goal_loc)

        grid[grid_tmp == 0] = np.inf
        path_points = pyastar.astar_path(grid, (current_loc[1], current_loc[0]), (goal_loc[1], goal_loc[0]), allow_diagonal=True)

        '''
        grid[grid_tmp == 0] = 255  # 65025
        grid[grid_tmp == 1] = 0  # 65025
        # grid[goal_loc[0], goal_loc[1]] = 0
        # grid[current_loc[0], current_loc[1]] = 0
        # print(grid[int(self.current_loc[0]-15):int(self.current_loc[0]+15), int(self.current_loc[1]-15):int(self.current_loc[1]+15)])
        # print(grid[current_loc[0]-2-5:current_loc[0]-2+5, current_loc[1]-2-5:current_loc[1]-2+5])

        pStart, pEnd = Point(current_loc[1], current_loc[0]), Point(goal_loc[1], goal_loc[0])
        aStar = AStar(grid, pStart, pEnd)
        aStar.expansion(offset=1)
        path_points = aStar.start()
        '''

        try:
            # path = []
            # for i in path_points:
            #     path.append([i.y, i.x])
            #
            # path = np.array(path)
            try:
                path = path_points[:, [1, 0]]
            except:
                pass
                # print(path)

            dist = 0
            position = current_loc
            # print('\n', path)
            # if path.shape[0] == 0:
            #     cv2.imwrite('grid.png', grid)

            for i in range(path.shape[0]):
                pos = path[i]
                if dist < 5:
                    previous_dist = dist
                    previous_position = position
                    position = pos
                    dist = np.linalg.norm(position - current_loc)
                else:
                    break

            if previous_dist <= dist:
                final_position = previous_position
            else:
                final_position = position

            final_relative = final_position - current_loc
            angle = int(np.rad2deg(np.arctan2(final_relative[1], final_relative[0])) // 10) * 10
            current_angle = np.rad2deg(self.current_loc[-1])

            map_real_angle = angle - current_angle  # - 20
            if not -180 < map_real_angle < 0:
                map_real_angle = (angle - current_angle) % 360

            if -5 < map_real_angle < 5:
                action = 0
            elif 5 <= map_real_angle < 180:
                action = 1  # turn left
            else:  # -180 <= angle <= 0:
                action = 2  # turn right

            if dist < 3:
                action = 3

            '''
            direction_x = current_loc[0] + int(10 * np.cos(np.deg2rad(current_angle)))
            direction_y = current_loc[1] + int(10 * np.sin(np.deg2rad(current_angle)))

            angent_direction_x = current_loc[0] + int(10 * np.cos(np.deg2rad(angle)))
            angent_direction_y = current_loc[1] + int(10 * np.sin(np.deg2rad(angle)))

            print('final_position: ', final_position, 'current_loc: ', current_loc, 'goal_loc: ', goal_loc,
                  'current_angle: ', current_angle, 'map_real_angle: ', map_real_angle)
            fig, axes = subplot(plt, (1, 1))
            ax = axes
            locs = path

            ax.imshow(grid, origin='lower')
            ax.plot(locs[:, 0], locs[:, 1], 'm.', ms=3)
            ax.plot(direction_x, direction_y, 'bx')
            ax.plot(angent_direction_x, angent_direction_y, 'rx')
            # if locs.shape[0] > 0:
            #     ax.plot(locs[0, 0], locs[0, 1], 'bx')
            ax.plot(current_loc[0], current_loc[1], 'b.')
            ax.plot(goal_loc[0], goal_loc[1], 'y*')

            plt.savefig(os.path.join('data/slam_result/out_dir_dstar/{0:01d}_{1:01d}_{2:.2f}_{3:.2f}_{4}_{5}.png'.format(step_num, action, current_angle, map_real_angle, str(current_loc), str(final_position))),
                        bbox_inches='tight')
            plt.close()
            '''
        except:
            # print(path_points)
            action = np.random.randint(3)
            '''
            print('path is None.')
            fig, axes = subplot(plt, (1, 1))
            ax = axes

            ax.imshow(grid, origin='lower')
            # if locs.shape[0] > 0:
            #     ax.plot(locs[0, 0], locs[0, 1], 'bx')
            ax.plot(current_loc[0], current_loc[1], 'b.')
            ax.plot(goal_loc[0], goal_loc[1], 'y*')

            plt.savefig(os.path.join(
                'data/slam_result/out_dir_dstar/{0:01d}_{1:01d}.png'.format(step_num, action)),
                        bbox_inches='tight')
            plt.close()
            '''

        return action
