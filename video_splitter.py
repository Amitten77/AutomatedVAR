import cv2
import os
vidcap = cv2.VideoCapture('mbappe_offsides.mov')
success,image = vidcap.read()
count = 0
if not os.path.isdir("./split_frames"): 
    main_dir = "./split_frames"
    os.mkdir(main_dir) 
while success:
  cv2.imwrite("./split_frames/frame%d.jpg" % (count), image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1


  #Include true negatives in dataset