import pixellib
from pixellib.semantic import semantic_segmentation
import cv2

def image_segmentation(string, number):
    segment_image = semantic_segmentation()
    segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5") 
    segment_image.segmentAsPascalvoc(string, output_image_name = "/Users/amit/Desktop/Sys_Senior_Project/temp_images/" + str(number) + "out.jpg")


