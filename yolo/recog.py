from turtle import width
import cv2
import numpy as np
import matplotlib.pyplot as plt

yolo = cv2.dnn.readNet(r"C:\Users\Saurav Banerjee\Documents\GitHub\AutomatedVAR\yolo\yolov3-tiny.weights", r"C:\Users\Saurav Banerjee\Documents\GitHub\AutomatedVAR\yolo\yolov3-tiny.cfg")
classes = []

with open(r"C:\Users\Saurav Banerjee\Documents\GitHub\AutomatedVAR\yolo\coco.names", 'r') as f:
    classes = f.read().splitlines()

img = cv2.imread("falcao.jpg")

blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0), swapRB=True, crop= False)
print(blob.shape)

#print image
i = blob[0].reshape(320,320,3)
plt.imshow(i)
plt.pause(5)

yolo.setInput(blob)
output_layers_name = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layers_name)

boxes = []
confidences = []
class_ids = []

width = 320
height = 320
for output in layeroutput:
    for detection in output:
        score = detection[:5]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.9:
            center_x = int(detection[0]*width)
            center_y = int(detection[0]*height)
            w = int(detection[0]*width)
            h = int(detection[0]*height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
print(len(boxes))

indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size = (len(boxes),4))

for i in indexes.flatten():
    x,y,w,h = boxes[i]

    label = str(classes[class_ids[i]])
    confi = str(round(confidences[i],2))
    color = colors[i]

    cv2.rectangle(img, (x,y),(x+w, y+h), color,1)
    cv2.putText(img, label + " " + confi, (x,y+20), font, 2,(255,255,255),1)
plt.imshow(img)
plt.pause(50)