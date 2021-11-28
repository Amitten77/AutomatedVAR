from models import *
from utils import *
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from kmeans import *
import os
import shutil
from image_segmentation import *
from line_detection import *
import config
import random

sys.setrecursionlimit(10**6)
#num = random.randint(0, 489)
num = config.num

the_play_initial = Image.open(r"./Offside_Images/" + str(num) + ".jpg")

the_play = the_play_initial.copy()

if not os.path.isdir("./temp_images"): 
    main_dir = "./temp_images"
    os.mkdir(main_dir) 

for i in range(the_play.size[0]):
    for j in range(the_play.size[1]):
        rgb = the_play.getpixel((i, j))
        if rgb[0] > 100 and rgb[1] > 150 and rgb[2] > 100:#red green blue
            the_play.putpixel((i, j), (255, 255, 255))
        else:
            the_play.putpixel((i, j), (0, 0, 0))
        

the_play.show()

rgb_im = the_play.convert("RGB")

rgb_im.save("./temp_images/" + str(num) + "blackandwhite.jpg")


lines = line_detection("./temp_images/" + str(num) + "blackandwhite.jpg")

lines.show()

def rgb_mean_distance(rgb1, rgb2):
    r1, g1, b1, = rgb1
    r2, g2, b2 = rgb2
    return pow((r2-r1), 2) + pow((g2-g1), 2) + pow((b2-b1), 2)

red = (255, 0, 0)
white = (255, 255, 255)
black = (0, 0, 0)

linesets = []
used_coords = set()


def fill(count, lines, x, y, index, color):#Area Fill!
    linesets[index].append((x, y))
    used_coords.add((x, y))
    lines.putpixel((x, y), color)
    if (x-1, y) not in used_coords and x - 1 != -1 and (rgb_mean_distance(lines.getpixel((x-1, y)), red) <  rgb_mean_distance(lines.getpixel((x-1, y)), black)) and (rgb_mean_distance(lines.getpixel((x-1, y)), red) <  rgb_mean_distance(lines.getpixel((x-1, y)), white)):
        fill(count + 1, lines, x-1, y, index, color)
    if (x+1, y) not in used_coords and x + 1 != lines.size[0] and (rgb_mean_distance(lines.getpixel((x+1, y)), red) <  rgb_mean_distance(lines.getpixel((x+1, y)), black)) and (rgb_mean_distance(lines.getpixel((x+1, y)), red) <  rgb_mean_distance(lines.getpixel((x+1, y)), white)):
        fill(count + 1, lines, x+1, y, index, color)
    if (x, y-1) not in used_coords and y - 1 != -1 and (rgb_mean_distance(lines.getpixel((x, y-1)), red) <  rgb_mean_distance(lines.getpixel((x, y-1)), black)) and (rgb_mean_distance(lines.getpixel((x, y-1)), red) <  rgb_mean_distance(lines.getpixel((x, y-1)), white)):
        fill(count + 1, lines, x, y-1, index, color)
    if (x, y+1) not in used_coords and y + 1 != lines.size[1] and (rgb_mean_distance(lines.getpixel((x, y+1)), red) <  rgb_mean_distance(lines.getpixel((x, y+1)), black)) and (rgb_mean_distance(lines.getpixel((x, y+1)), red) <  rgb_mean_distance(lines.getpixel((x, y+1)), white)):
        fill(count + 1, lines, x, y+1, index, color)




for i in range(lines.size[0]):
    for j in range(lines.size[1]):
        rgb = lines.getpixel((i, j))
        if (i, j) not in used_coords and rgb_mean_distance(rgb, red) < rgb_mean_distance(rgb, white) and rgb_mean_distance(rgb, red) < rgb_mean_distance(rgb, black):
            linesets.append([])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            #color = (0, 0, 255)
            fill(0, lines, i, j, len(linesets)-1, color)


lines.show()


shutil.rmtree("./temp_images")





















