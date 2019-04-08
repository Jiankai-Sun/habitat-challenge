import json
from pprint import pprint

json_file = '/home/SENSETIME/sunjiankai/data/Program/habitat-challenge/habitat-challenge-data/gibson.val_mini.json'
# json_file = 'val.json'

scene_id_collections = []
with open(json_file, 'r') as f:
    data = json.load(f)

pprint(data['episodes'][0])
# pprint(data['episodes'])
pprint(len(data['episodes']))

for each_episode in data['episodes']:
    scene_id = each_episode['scene_id']
    if not (scene_id in scene_id_collections):
        scene_id_collections.append(scene_id)

print('len(scene_id_collections): {}\n'.format(len(scene_id_collections)))
print(scene_id_collections)