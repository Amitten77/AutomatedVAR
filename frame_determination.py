from ftplib import parse229
from re import M, S
from turtle import distance
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
import config
import random
import numpy as np
#from person_detection import *


#os.system("python3 person_detection.py --input ./mbappe_offsides.mov --output mbappe_offsides_detections.mov --yolo config")

x_list = []
soccer_ball_coordinates = []
with open("soccer_ball.txt") as f:
    lines = f.readlines()
    count = -1
    for line in lines:
        line = line.strip()
        count += 1
        if line != "X":
            w = line.split(",")
            num1 = int(float((w[0][1:])))
            num2 = int(float((w[1][1:-1])))
            soccer_ball_coordinates.append((num1, num2))
        else:
            x_list.append(count)
            soccer_ball_coordinates.append("X")



#print(soccer_ball_coordinates)
person_dic = {}
count = 1
with open("player_positions.txt") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if "," not in line and len(line) != 0:
            count = int(line)
            person_dic[count] = []
        elif(len(line) != 0):
            w = line.split(",")
            num1 = int(float((w[0][1:])))
            num2 = int(float((w[1][1:-1])))
            person_dic[count].append((num1, num2))


def closest_point(lis, point):
    min_distance = 9999999999
    best_point = "X"
    x, y = point
    for i in range(len(lis)):
        a, b = lis[i]
        total = pow((a-x), 2) + pow((b-y), 2)
        distance = pow(total, 1/2)
        if distance < min_distance:
            min_distance = distance
            best_point = (a,b)
    return best_point

def surrounding_points(point):
    lis = []
    a, b = point
    lis.append((a-1, b-1))
    lis.append((a, b-1))
    lis.append((a+1, b-1))
    lis.append((a-1, b))
    lis.append((a, b))
    lis.append((a+1, b))
    lis.append((a-1, b+1))
    lis.append((a, b+1))
    lis.append((a+1, b+1))
    return lis



myImg = Image.open("./split_frames/frame0.jpg")
pixels = myImg.load()
'''
for i in range(len(person_dic[1])):
    a,b = person_dic[1][i]
    surrounding_pixels_player = surrounding_points((a,b))
    for j in range(9):
        p1, p2 = surrounding_pixels_player[j]
        if p2 < myImg.size[1]:
            pixels[p1, p2] = (255, 0, 0)
myImg.show()


'''





a, b = closest_point(person_dic[1], soccer_ball_coordinates[0])

coordinates_of_passing_player = [(a, b)]
most_recent = (a,b)
for key in person_dic:
    if key > 1 and key != len(person_dic):
        p1, p2 = closest_point(person_dic[key], most_recent)
        coordinates_of_passing_player.append((p1, p2))
        most_recent = (p1, p2)


#print(x_list)
#print(soccer_ball_coordinates)

#This code is what helps fill in the gaps in the soccer ball coordinates
#This is a list of each gap: the indices before and after
gaps = []
previous = x_list[0]
g1 = x_list[0]
for i in range(len(x_list)):
    if i != 0: 
        if x_list[i] == previous + 1:
            previous += 1
        else:
            g2 = previous
            gaps.append((g1 - 1, g2 + 1))
            g1 = x_list[i]
            previous = x_list[i]

for gap in gaps:
    start, end = gap
    gap_len = end-start
    diff1 = soccer_ball_coordinates[end][0] - soccer_ball_coordinates[start][0]
    diff2 = soccer_ball_coordinates[end][1] - soccer_ball_coordinates[start][1]
    add1 = float(float(diff1)/float(gap_len))
    add2 = float(float(diff2)/float(gap_len))
    for i in range(start + 1, end):
        a = int(soccer_ball_coordinates[i-1][0] + add1)
        b = int(soccer_ball_coordinates[i-1][1] + add2)
        soccer_ball_coordinates[i] = (a,b)
'''
print(len(soccer_ball_coordinates))
print()
print(len(coordinates_of_passing_player))
'''
distance_lis = []
for i in range(len(soccer_ball_coordinates)): #gets the distance between the player and the ball
    if(soccer_ball_coordinates[i] != "X"):
        d1 = soccer_ball_coordinates[i][0] - coordinates_of_passing_player[i][0]
        d2 = soccer_ball_coordinates[i][1] - coordinates_of_passing_player[i][1]
        s = pow(d1, 2) + pow(d1, 2)
        distance_lis.append(pow(s, 1/2))
    else:
        distance_lis.append("N/A")

print(distance_lis)
previous_value = 0
the_frame = -1
count = 0
for i in range(len(distance_lis)):
    if i == 0:
        previous_value = distance_lis[i]
    else:
        if abs(distance_lis[i]-previous_value>5) and count == 0:
            possible_end = True
            the_frame = i-1
            count += 1
        if count > 0:
            if abs(distance_lis[i] - previous_value>5):
                count += 1
            if count == 6:
                break
        else:
            previous_value = distance_lis[i]

if the_frame != -1:
    myImg = Image.open("./split_frames/frame" + str(the_frame) + ".jpg")
    myImg.save("final_frame.jpg")
'''
for i in range(130):
    myImg = Image.open("./split_frames/frame" + str(i) + ".jpg")
    pixels = myImg.load()
    surround = surrounding_points(coordinates_of_passing_player[i])
    for j in range(9):
        p1, p2 = surround[j]
        pixels[p1, p2] = (255, 0, 0)
    myImg.save("./player_tracking/frame" + str(i) + ".jpg")
'''
'''
myImg = Image.open("./split_frames/frame0.jpg")
pixels = myImg.load()
length =  myImg.size[0]
height = myImg.size[1]
surrounding_pixels_player = surrounding_points((a,b))
surrounding_pixels_ball = surrounding_points((soccer_ball_coordinates[0]))
for i in range(9):
    p1, p2 = surrounding_pixels_ball[i]
    p3, p4 = surrounding_pixels_player[i]
    pixels[p1, p2] = (255, 0, 0)
    pixels[p3, p4] = (0, 255, 0)
myImg.show()
'''


#predict where soccer ball will go



#In frame 1, find player closest to the ball
#Wait until ball leaves 


