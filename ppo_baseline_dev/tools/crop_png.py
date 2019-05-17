from PIL import Image
import os

INPUT_DIR = './out_dir_0'
OUTPUT_DIR = './top_down'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for i in range(999):
    im = Image.open(os.path.join(INPUT_DIR, '{:04d}.png'.format(i)))
    print(im.getbbox())
    im2 = im.crop((47, 35, 768, 763))
    im2.save(os.path.join(OUTPUT_DIR, '{:04d}.png'.format(i)))
