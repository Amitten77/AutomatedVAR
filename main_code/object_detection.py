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
import config
import goaline_detection
from kmeans import k_means_dic
from kmeans import k_means
import math


#num = random.randint(0, 489)
num = config.num
goal_facing = "RIGHT" #Hard-coded for now, change later

#the_play = "./Offside_Images/" + str(num) + ".jpg"
the_play = "./final_frame.jpg"

if not os.path.isdir("./temp_images"): 
    main_dir = "./temp_images"
    os.mkdir(main_dir) 



config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.7
nms_thres=0.4
# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
#model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.FloatTensor


def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0), 
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, 
                        conf_thres, nms_thres)
    return detections[0]


    # load image and get detections
img_path = the_play
prev_time = time.time()
img = Image.open(img_path)
#print(img.size[0], img.size[1])
detections = detect_image(img)
inference_time = datetime.timedelta(seconds=time.time() - prev_time)
#print ('Inference Time: %s' % (inference_time))
# Get bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
img = np.array(img)
plt.figure()
fig, ax = plt.subplots(1, figsize=(12,9))
ax.imshow(img)
pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
unpad_h = img_size - pad_y
unpad_w = img_size - pad_x
if detections is not None:
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)
    # browse detections and draw bounding boxes
    zoom_in = []
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
        zoom_in.append([float(x1), float(y1), float(x1+box_w), float(y1+box_h)])
        color = bbox_colors[int(np.where(
             unique_labels == int(cls_pred))[0])]
        bbox = patches.Rectangle((x1, y1), box_w, box_h,
             linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox)
        plt.text(x1, y1, s=classes[int(cls_pred)], 
                color='white', verticalalignment='top',
                bbox={'color': color, 'pad': 0})
plt.axis('off')
# save image
plt.savefig(img_path.replace(".jpg", "-det.jpg"),        
                  bbox_inches='tight', pad_inches=0.0)
plt.show()


#print(zoom_in)
'''
#(rgb)
#Red Color Range: [127-255, 0-77, 0-77]

#Blue Color Range: [0-77 , 0-77, 127-255]

#Green Color Range: [0-77, 127-255, 0-77]

'''
def pixel_color(color_dict, rgb):
    r, g, b = rgb
    for key in color_dict:
        if r >= color_dict[key][0][0] and r <= color_dict[key][1][0]:
            if g >= color_dict[key][0][1] and g <= color_dict[key][1][1]:
                if b >= color_dict[key][0][2] and g <= color_dict[key][1][2]:
                    return key
    return "NONE"

def rgb_mean_distance(rgb1, rgb2):
    r1, g1, b1, = rgb1
    r2, g2, b2 = rgb2
    return pow((r2-r1), 2) + pow((g2-g1), 2) + pow((b2-b1), 2)



im1 = Image.open(the_play)


color_dict = {}
color_dict["RED"] = ((127, 0, 0), (255, 77, 77))
color_dict["GREEN"] = ((0, 0, 127), (77, 77, 255))
color_dict["BLUE"] = ((0, 127, 0), (77, 255, 77))





x_vals = []
for i in range(len(zoom_in)):
    x_vals.append((zoom_in[i][0] + zoom_in[i][2]/2))


goalkeeper = x_vals.index(max(x_vals))


#zoom_in: [x1, y1, x2, y2]
print(zoom_in)
print(x_vals, goalkeeper)

#sys.exit()
#Color Approach:

all_detections = []

pixel_dict = {}
error_dict = {}
pink = (192, 128, 128)
black = (0, 0, 0)
field_green = (120, 180, 0)
for i in range(len(zoom_in)): #zooms in on each person
    pixel_dict[i] = [0, 0, 0]
    total = 0
    im3 = im1.crop(zoom_in[i])#crops the picture so it's just the box that selects the person
    im3.save("./temp_images/" + str(i) + ".jpg", "jpeg")
    image_segmentation("./temp_images/" + str(i) + ".jpg", i)
    im3 = Image.open("./temp_images/" + str(i) + ".jpg")
   # im3.show()
    new_dic = k_means(2, im3)
    correct_key = "x"
    max_distance = 0
    for key in new_dic:
        num = rgb_mean_distance(key, field_green)
        if num > max_distance:
            correct_key = key
            max_distance = num
    for j in range(im3.size[0]):
        for k in range(im3.size[1]):
            rgb3 = im3.getpixel((j, k))    
            if rgb3 in new_dic[correct_key]: #if pixel is actually part of the person
                pixel_dict[i][0] += rgb3[0]
                pixel_dict[i][1] += rgb3[1]
                pixel_dict[i][2] += rgb3[2]
                total += 1
    pixel_dict[i][0] = pixel_dict[i][0]/total
    pixel_dict[i][1] = pixel_dict[i][1]/total
    pixel_dict[i][2] = pixel_dict[i][2]/total
    im4 = Image.open("./temp_images/" + str(i) + "out.jpg")
    thergb = [0, 0, 0]
    for j in range(im4.size[0]):
        for k in range(im4.size[1]):
            rgb4 = im4.getpixel((j, k))    
            if rgb_mean_distance(rgb4, pink) < rgb_mean_distance(rgb4, black): #if pixel is actually part of the person
                rgb2 = im3.getpixel((j, k))
                thergb[0] += rgb2[0]
                thergb[1] += rgb2[1]
                thergb[2] += rgb2[2]
                total += 1
    thergb[0] = thergb[0]/total
    thergb[1] = thergb[1]/total
    thergb[2] = thergb[2]/total
    pixel_dict[i][0] = (pixel_dict[i][0] + thergb[0])/2
    pixel_dict[i][1] = (pixel_dict[i][1] + thergb[1])/2
    pixel_dict[i][2] = (pixel_dict[i][2] + thergb[2])/2

if goalkeeper - 1 > -1:
    pixel_dict[goalkeeper] = pixel_dict[goalkeeper-1]

final_dic = k_means_dic(2, pixel_dict)



cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
im1 = np.array(im1)
plt.figure()
fig, ax = plt.subplots(1, figsize=(12,9))
ax.imshow(im1)
for i in range(len(zoom_in)):
        x1 = zoom_in[i][0]
        y1 = zoom_in[i][1]
        box_w = zoom_in[i][2] - x1
        box_h = zoom_in[i][3] - y1
        color = bbox_colors[int(np.where(
                unique_labels == int(cls_pred))[0])]
        bbox = patches.Rectangle((x1, y1), box_w, box_h,
                linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox)
        key = tuple(pixel_dict[i])
        category = final_dic[key]
        if i != goalkeeper:
            if category == 0:
                plt.text(x1, y1, s="Team 1, Player #" + str(i), 
                        color='white', verticalalignment='top',
                        bbox={'color': color, 'pad': 0})
            elif category == 1:
                plt.text(x1, y1, s="Team 2, Player #" + str(i), 
                        color='white', verticalalignment='top',
                        bbox={'color': color, 'pad': 0})
        else:
            plt.text(x1, y1, s="Not Sure or Other", 
                    color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})
    

plt.axis('off')
plt.savefig("linetestcase.png", bbox_inches='tight', pad_inches = 0)
plt.show()


def draw_line(slope, point, image_string, color, output_image_string):
    img = Image.open(image_string)
    w, x = point
    y, z = point
    while round(w) < img.size[0] and round(x) < img.size[1]:
        img.putpixel((round(w), round(x)), color)
        w += 1
        x += slope
    y -= 1
    z -= slope
    y -= 1
    z -= slope
    while round(y) > 0 and round(z) > 0:
        img.putpixel((round(y), round(z)), color)
        y -= 1
        z -= slope
    img.save(output_image_string)


#zoom_in = [[60.91776657104492, 423.3991394042969, 117.04763793945312, 561.062744140625], [1014.1339721679688, 523.4462890625, 1086.634521484375, 671.8738403320312], [590.0214233398438, 918.8041381835938, 691.9923095703125, 1083.85595703125], [978.5548706054688, 428.4405212402344, 1043.1126708984375, 542.8153686523438], [663.6605224609375, 268.0061340332031, 699.7683715820312, 345.74102783203125], [877.8689575195312, 448.444580078125, 938.8426513671875, 571.3521728515625], [499.7018127441406, 296.6759338378906, 567.9681396484375, 404.9012756347656], [1279.5833740234375, 340.46588134765625, 1333.8079833984375, 454.10650634765625]]
num = config.num
val = int(input("Enter the player number of the player who recieved the ball?\n"))

point = zoom_in[int(val)]
point = (point[2], point[3])
#os.system("python3 goaline_detection.py")
slope = (2 * goaline_detection.m + goaline_detection.out_m)/2
y_intercept = goaline_detection.b
draw_line(slope, point, "./final_frame.jpg", (0, 0, 255), "./thetest.jpg")

def distance_point_to_line(x1, y1, a, b, c):#Ax + By + c = 0
    d = abs(((-a) * x1 + b * y1 - c)) / (math.sqrt(a * a + b * b))
    return d



first_team = final_dic[list(final_dic.keys())[val]]
last_man_back = 100000
count = 0
min_distance = 100000000
for key in final_dic:
    if str(final_dic[key]) != str(first_team) and count != goalkeeper:
        x = zoom_in[count][2]
        y = zoom_in[count][3]
        d = distance_point_to_line(x, y, slope, 1, y_intercept)
        if d < min_distance:
            min_distance = d
            last_man_back = count
    count += 1
    

#last_man_back = 6
point = zoom_in[int(last_man_back)]
point = (point[2], point[3])

draw_line(slope, point, "./thetest.jpg", (255, 0, 0), "./thetest.jpg")

img = Image.open("./thetest.jpg")
img.show()



#shutil.rmtree("./temp_images")