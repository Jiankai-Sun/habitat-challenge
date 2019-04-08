import dominate
from dominate.tags import *
import os

top_down_vis_dir = 'top_down_vis'
video_dir = 'video'
image_file_list = sorted(os.listdir(top_down_vis_dir))[:10]

web_path = ''
with dominate.document(title=web_path) as web:
    for each_file in image_file_list:
        file_name = each_file.split('.')[0]
        success = file_name.split('_')[1]

        h2('{0}'.format(file_name))

        with table(border=1, style='table-layout: fixed;'):
            with tr():
                with tr(style='word-warp: break-world;', halign='center', valign='top'):
                    with td(style='word-warp: break-world;', halign='center', valign='top'):
                        img(style='width:128px', src='{}.png'.format(os.path.join(top_down_vis_dir, file_name)))
                    with td(style='word-warp: break-world;', halign='center', valign='top'):
                        video(style='width:128px',
                              src='{}.mp4'.format(os.path.join(video_dir, file_name), controls="controls",
                                                  autoplay="autoplay"))

with open(os.path.join(web_path, 'index.html'), 'w') as fp:
    fp.write(web.render())
