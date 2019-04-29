import numpy as np
import os
from absl import app
from yattag import Doc
from yattag import indent
import json

# OUT_DIR="../data/slam_result/out_dir_0"
OUT_DIR="../data/slam_result/out_dir_num_erosion_2_mapsize_2400/"
JSON_FILE_NAME = '/data/Dataset/habitat-api-data/datasets/pointnav/gibson/v1/val/val.json'

with open(JSON_FILE_NAME, 'rb') as f:
  json_data = json.load(f)

def main(_):
  doc, tag, text = Doc().tagtext()
  spl_file = os.path.join(OUT_DIR, 'spls.txt')
  spls = np.loadtxt(spl_file, delimiter=' ')
  inds = np.argsort(spls[:,0])
  with tag('html'):
    with tag('body'):
      with tag('table'):
        scene_name_list = list()
        geodesic_distance_list = list()
        index_list = list()
        spl_list = list()
        last_name = None
        for i in inds[:len(json_data['episodes'])]:
          scene_name = json_data['episodes'][i % 994]['scene_id'].split('/')[-1].split('.')[0]
          geodesic_distance = json_data['episodes'][i % 994]['info']['geodesic_distance']

          if last_name == scene_name:
            scene_name_list.append(last_name)
            geodesic_distance_list.append(geodesic_distance)
            index_list.append(i)
            spl_list.append(spls[i,1])
          elif last_name != scene_name and last_name is not None:
            scene_name_list = [x for (y,x) in sorted(zip(geodesic_distance_list, scene_name_list))]
            index_list = [x for (y,x) in sorted(zip(geodesic_distance_list, index_list))]
            spl_list = [x for (y,x) in sorted(zip(geodesic_distance_list, spl_list))]
            geodesic_distance_list = sorted(geodesic_distance_list)

            print(index_list)
            with tag('tr'):
              for j in range(len(scene_name_list)):
                with tag('th'):
                  text('{0:04d} - {1:0.4f} - {2} - {3}'.format(index_list[j], spl_list[j], scene_name_list[j], geodesic_distance_list[j]))
            with tag('tr'):
              for j in range(len(scene_name_list)):
                with tag('th'):
                  with tag('img', src=os.path.join('..', 'top_down', '{:04d}.png'.format(index_list[j])), height='256px', ):
                    None
            scene_name_list = list()
            geodesic_distance_list = list()
            index_list = list()
            spl_list = list()
          last_name = scene_name

  out_file = os.path.join(OUT_DIR, 'global_map.html')
  with open(out_file, 'w') as f:
    print(indent(doc.getvalue()), file=f)
  # good_inds = list(range(682)) + list(range(744, 1000))
  print('Mean success rate: {0}'.format(np.mean(spls[:,1] > 0)))
  print('Mean SPL: {0}'.format(np.mean(spls[:,1])))
  # print(np.mean(spls[good_inds,1]))
  # print(np.mean(spls[good_inds,1] > 0))

if __name__ == '__main__':
  app.run(main)
