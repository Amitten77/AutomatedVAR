from PIL import Image
from matplotlib import cm
import numpy as np
import cv2


#path = '196.jpg'
#img = cv2.imread(path)
def k_means(img):

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    twoDimage = img.reshape((-1,3))
    twoDimage = np.float32(twoDimage)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    attempts=10

    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    PIL_image = Image.fromarray(np.uint8(result_image)).convert('RGB')

    PIL_image.show()