import os
import numpy as np 
import pandas as pd
import shutil as sh
from PIL import Image
from tqdm.auto import tqdm

data_path = "/Users/amit/Desktop/AutomatedVAR"
df = pd.read_csv('./via_export_csv.csv')
## create x, y, w, h columns 
x, y, w, h = [], [], [], []
for row in df['region_shape_attributes']:
    row = row.replace('{}', '').replace('}', '')
    row = row.split(',')
    x.append(int(row[1].split(':')[-1]))
    y.append(int(row[2].split(':')[-1]))
    w.append(int(row[3].split(':')[-1]))
    h.append(int(row[4].split(':')[-1]))
## calculating x, y, width and height coordinates
df['x'], df['y'], df['w'], df['h'] = x, y, w, h
## creating a column name image_id having images names as id 
df['image_id'] = [name.split('.')[0] for name in df['filename']]
## creating two columns for storing x and y center values
df['x_center'] = df['x'] + df['w']/2
df['y_center'] = df['y'] + df['h']/2
## define number of classes 
labels = df['region_attributes'].unique()
labels_to_dict = dict(zip(labels, range(0, len(labels))))
print('Lables Directory:', labels_to_dict)
df['classes'] = df['region_attributes']
df.replace({'classes':labels_to_dict}, inplace=True)
df = df[['image_id','x', 'y', 'w', 'h','x_center','y_center','classes']]
## set index of images
index = list(set(df.image_id))


source = 'split_frames'
if True:
    for fold in [0]:
        val_index = index[len(index)*fold//5:len(index)*(fold+1)//5]
        for name,mini in tqdm(df.groupby('image_id')):
            if name in val_index:
                path2save = 'val2017/'
            else:
                path2save = 'train2017/'
            if not os.path.exists('convertor/fold{}/labels/'.format(fold)+path2save):
                os.makedirs('convertor/fold{}/labels/'.format(fold)+path2save)
            with open('convertor/fold{}/labels/'.format(fold)+path2save+name+".txt", 'w+') as f:
                row = mini[['classes','x_center','y_center','w','h']].astype(float).values
                imagename = data_path+"/{}/{}.jpg".format(source,name)
                check_image_width_height = Image.open(imagename)
                img_width, img_height = check_image_width_height.size
                for r in (row):
                    r[1] = r[1]/img_width
                    r[2] = r[2]/img_height
                    r[3] = r[3]/img_width
                    r[4] = r[4]/img_height
                row = row.astype(str)
                for j in range(len(row)):
                    print(row[j], 'n')
                    row[j][0] = str(int(float(row[j][0])))
                    text = ' '.join(row[j])
                    f.write(text)
                    f.write("n")
            if not os.path.exists('convertor/fold{}/images/{}'.format(fold,path2save)):
                os.makedirs('convertor/fold{}/images/{}'.format(fold,path2save))
            sh.copy(data_path+"/{}/{}.jpg".format(source,name),'convertor/fold{}/images/{}/{}.jpg'.format(fold,path2save,name))


#python3 train.py --img 416 --batch 12 --epochs 10 --data ./data/cocov2.yaml --weights ./weights/yolov5x.pt
#python3 detect.py --img 416 --source ./frame30.jpg --weights ./runs/train/exp9/weights/best.pt --conf-thres 0.02