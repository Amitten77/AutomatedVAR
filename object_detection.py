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


#num = random.randint(0, 489)
num = 113


the_play = "./Offside_Images/" + str(num) + ".jpg"


if not os.path.isdir("./temp_images"): 
    main_dir = "./temp_images"
    os.mkdir(main_dir) 



config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
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
#plt.show()


#print(zoom_in)
'''
(rgb)
Red Color Range: [127-255, 0-77, 0-77]

Blue Color Range: [0-77 , 0-77, 127-255]

Green Color Range: [0-77, 127-255, 0-77]

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



#Color Approach:

all_detections = []

pixel_dict = {}
error_dict = {}
pink = (192, 128, 128)
black = (0, 0, 0)
standard = (0, 0 , 0)
for i in range(len(zoom_in)):
    pixel_dict[i] = [0, 0, 0]
    total = 0
    im2 = im1.crop(zoom_in[i])
    im2.save("./temp_images/" + str(i) + ".jpg", "jpeg")
    image_segmentation("./temp_images/" + str(i) + ".jpg", i)
    im3 = Image.open("./temp_images/" + str(i) + ".jpg")
    for j in range(im3.size[0]):
        for k in range(im3.size[1]):
            rgb3 = im3.getpixel((j, k))    
            if rgb_mean_distance(rgb3, pink) < rgb_mean_distance(rgb3, black): #if pixel is actually part of the person
                rgb2 = im2.getpixel((j, k))
                pixel_dict[i][0] += rgb2[0]
                pixel_dict[i][1] += rgb2[1]
                pixel_dict[i][2] += rgb2[2]
                total += 1
    pixel_dict[i][0] = pixel_dict[i][0]/total
    pixel_dict[i][1] = pixel_dict[i][1]/total
    pixel_dict[i][2] = pixel_dict[i][2]/total
    if i == 0:
        standard = pixel_dict[i]
    error_dict[i] = rgb_mean_distance(standard, pixel_dict[i])


print(error_dict)



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
    if error_dict[i] < 3000:
        plt.text(x1, y1, s="Team 1", 
                color='white', verticalalignment='top',
                bbox={'color': color, 'pad': 0})
    else:
        plt.text(x1, y1, s="Team 2", 
                color='white', verticalalignment='top',
                bbox={'color': color, 'pad': 0})
    '''
    else:
        plt.text(x1, y1, s="Referee/Goalkeeper", 
                color='white', verticalalignment='top',
                bbox={'color': color, 'pad': 0})
    '''

plt.axis('off')
plt.show()




shutil.rmtree("./temp_images")