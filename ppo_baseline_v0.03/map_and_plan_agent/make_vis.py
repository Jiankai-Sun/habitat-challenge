import numpy as np
import os
from absl import app
from yattag import Doc
from yattag import indent

OUT_DIR="/data/sunjiankai/Dataset/habitat_data/slam_result/out_dir_multi_step_2"

def main(_):
  doc, tag, text = Doc().tagtext()
  spl_file = os.path.join(OUT_DIR, 'spls.txt')
  spls = np.loadtxt(spl_file, delimiter=' ')
  inds = np.argsort(spls[:,1])
  with tag('html'):
    with tag('body'):
      with tag('table'):
        for i in inds[:50]:
          with tag('tr'):
            with tag('td'):
              text('{:04d} - {:0.4f}'.format(i, spls[i,1]))
            with tag('td'):
              with tag('img', src='{:04d}_rgbs_depths_fmms_maps.gif'.format(i)):
                None
            # with tag('td'):
            #   with tag('img', src='{:04d}.gif'.format(i)):
            #     None
            # with tag('td'):
            #   with tag('img', src='{:04d}_d.gif'.format(i)):
            #     None
            # with tag('td'):
            #   with tag('img', src='{:04d}_map.gif'.format(i)):
            #     None
            # with tag('td'):
            #   with tag('img', src='{:04d}_fmm.gif'.format(i)):
            #     None
            with tag('td'):
              with tag('img', src='{:04d}.png'.format(i), height='256px'):
                None

  out_file = os.path.join(OUT_DIR, 'vis.html')
  with open(out_file, 'w') as f:
    print(indent(doc.getvalue()), file=f)
  # good_inds = list(range(682)) + list(range(744, 1000))
  print('Mean success rate: {0}'.format(np.mean(spls[:,1] > 0)))
  print('Mean SPL: {0}'.format(np.mean(spls[:,1])))
  # print(np.mean(spls[good_inds,1]))
  # print(np.mean(spls[good_inds,1] > 0))

if __name__ == '__main__':
  app.run(main)
