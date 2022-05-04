from black import out
from models import *
from utils import *
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from kmeans import *
import os
import shutil
from image_segmentation import *
from line_detection import *
import config
import random
import numpy as np
import cv2

sys.setrecursionlimit(10**6)
#num = random.randint(0, 489)
num = config.num
goal_facing = "RIGHT" #Hard-coded for now, change later

the_play_initial = Image.open("final_frame.jpg")

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
        

#the_play.show()

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



lines.save("./temp_images/lines.jpg")

longest_lines = []
potential_goal_lines = []
avgxs = []
m_and_b = []
#lines is our current image with all the lines
for i in range(len(linesets)):
    xlis = []
    ylis = []
    for j in range(len(linesets[i])):
        xlis.append(linesets[i][j][0])
        ylis.append(linesets[i][j][1])
    x = np.array(xlis)
    y = np.array(ylis)
    m, b = np.polyfit(x, y, 1)
    m_and_b.append((m, b))
    templines = lines.copy()
    total_count = 0
    on_line = 0
    orix = 0
    oriy = b
    x_values = 0
    while oriy < 0 or oriy > lines.size[1]:
        oriy += m
        orix += 1
    while orix < lines.size[0] and oriy < lines.size[1]:
        templines.putpixel((round(orix), round(oriy)), (0, 255, 0))
        x_values += round(orix)
        if (round(orix), round(oriy)) in used_coords:
            on_line += 1
        total_count += 1
        orix += 1
        oriy += m
    avgxs.append(x_values/total_count)
    longest_lines.append(on_line/total_count)
    if on_line/total_count > .4:
        d1 = ImageDraw.Draw(templines)
        d1.text((28, 36), str(on_line/total_count), fill=(255, 0, 0))
        templines.save("./temp_images/templines" + str(i) + ".jpg")
        potential_goal_lines.append(i)



def distance_point_to_line(x1, y1, a, b, c):#Ax + By + c = 0
    d = abs(((-a) * x1 + b * y1 - c)) / (math.sqrt(a * a + b * b))
    return d

corner = 0
if goal_facing == "RIGHT":
    corner = [lines.size[0], 0]
else:
    corner = [0, lines.size[1]]


print(potential_goal_lines)
goal_line = potential_goal_lines[-2]
print(goal_line)
'''
if goal_facing == "RIGHT":
    extreme_x = -1
else:
    extreme_x = 99999
if len(potential_goal_lines) > 1:
    for index in potential_goal_lines:
        the_image = Image.open(r"./temp_images/templines" + str(index) + ".jpg")
        the_image.show()
        if goal_facing == "RIGHT" and avgxs[index] > extreme_x:
                extreme_x = avgxs[index]
                goal_line = index
        elif goal_facing == "LEFT" and avgxs[index] < extreme_x:
            extreme_x = avgxs[index]
            goal_line = index

min_distance = 99999999
max_distance = -3
outer_box_line = potential_goal_lines[1]
if len(potential_goal_lines) > 1:
    for index in potential_goal_lines:
        dist = distance_point_to_line(corner[0], corner[1], m_and_b[index][0], 1, m_and_b[index][1])
        if dist < min_distance:
            min_distance = dist
            goal_line = index
        if dist > max_distance:
            max_distance = dist
            outer_box_line = index
'''
outer_box_line = goal_line        

out_m, out_b = m_and_b[outer_box_line]


actual_image = Image.open("final_frame.jpg")
imagee = actual_image.copy()
m, b = m_and_b[goal_line]
orix = 0
oriy = b

while oriy < 0 or oriy > imagee.size[1]:
    oriy += m
    orix += 1
while orix < imagee.size[0] and oriy < imagee.size[1]:
    imagee.putpixel((round(orix), round(oriy)), (255, 0, 0))
    orix += 1
    oriy += m


print(m, b)
print(actual_image.size)

'''
if goal_facing == "RIGHT":
    m += 299000/(imagee.size[0]*imagee.size[1])
else:
    m -= 299000/(imagee.size[0]*imagee.size[1])
a = 10
b = 400
while a < imagee.size[0] and oriy < imagee.size[1]:
    imagee.putpixel((round(a), round(b)), (255, 0, 0))
    a += 1
    b += m
'''



imagee.show()


#shutil.rmtree("./temp_images")





















