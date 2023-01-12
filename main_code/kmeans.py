from PIL import Image
from matplotlib import cm
import numpy as np
import cv2
import random


#path = '196.jpg'
#img = cv2.imread(path)
'''
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
'''

def pick_pixel_lis(rgb, pix_set):#picks the rgb value from the pixel set that is closest to the random rgb value
    min_dist = 1000000000000000
    rgb1, rgb2, rgb3 = rgb#gets rgb value of current pixel
    answer = 0
    for pix in pix_set:
        temp = 0
        num1, num2, num3 = pix
        temp += (num1 - rgb1) ** 2
        temp += (num2 - rgb2) ** 2
        temp += (num3 - rgb3) ** 2
        if temp < min_dist:
            min_dist = temp
            answer = pix
    return answer


def k_means(k, img):
    pixel_set = set()
    pix3 = img.load()
    while len(pixel_set) < k: #picks K num pixels
        num1 = random.randint(0, img.size[0] - 1)
        num2 = random.randint(0, img.size[1] - 1)
        rgb = pix3[num1, num2]
        if rgb not in pixel_set:
            pixel_set.add(rgb)
    num_dic = {}
    count = 0
    pix_dic = {}#The dictionary that you need!
    for pix in pixel_set:
        num_dic[count] = pix
        count += 1
    dic = {} 
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            color = pix3[i, j]
            if color not in pix_dic:
                pix_dic[color] = 1
            else:
                pix_dic[color] += 1
            answer = pick_pixel_lis(color, pixel_set)
            if answer not in dic:
                dic[answer] = [color]
            else:
                dic[answer].append(color)
    gen = []
    gen.append(dic) #each generation of k means and their corresponding dictionary
    curr_gen = 0
    count = 0
    while(count < 3):
        old_lens = []
        new_dic = {}
        new_set = set()
        for i in range(k):
            totalr = 0
            totalg = 0
            totalb = 0
            total = 0
            for color in gen[curr_gen][num_dic[i]]:
                r, g, b = color
                totalr += r * pix_dic[color]
                totalg += g * pix_dic[color]
                totalb += b * pix_dic[color]
                total += pix_dic[color]
            old_lens.append(total)
            new_rgb = (totalr/total, totalg/total, totalb/total)
            new_set.add(new_rgb)
            num_dic[i] = new_rgb
        for color in pix_dic:
            answer = pick_pixel_lis(color, new_set)
            if answer not in new_dic:
                new_dic[answer] = [color]
            else:
                new_dic[answer].append(color)  
        gen.append(new_dic)
        moves = []
        for i in range(k):
            total = 0
            for color in new_dic[num_dic[i]]:
                total += pix_dic[color]
            moves.append(total - old_lens[i])
        curr_gen += 1
    #  print("Differences in gen " + str(curr_gen) + " : " + str(moves))
        for i in range(k):
            if moves[i] != 0:
                break
            if i == k - 1:
                count += 1
    return new_dic



def k_means_dic(k, lis_dic):
    lis = []
    for key in lis_dic:
        lis.append(tuple(lis_dic[key]))
    original_means = random.sample(lis, k)
    dic = {}
    for item in lis:
        answer = pick_pixel_lis(item, original_means)
        if answer not in dic:
            dic[answer] = [item]
        else:
            dic[answer].append(item)
    gen = []
    gen.append(dic)
    current_gen = 0
    count = 0
    while(count < 3):
        old_lens = []
        new_dic = {}
        new_set = set()
        for i in range(k):
            totalr = 0
            totalg = 0
            totalb = 0
            total = 0
            key = list(gen[current_gen].keys())[i]
            for color in gen[current_gen][key]:
                r, g, b = color
                totalr += r
                totalg += g
                totalb += b
                total += 1
            old_lens.append(total)
            new_rgb = (totalr/total, totalg/total, totalb/total)
            new_set.add(new_rgb)
        for color in lis:
            answer = pick_pixel_lis(color, new_set)
            if answer not in new_dic:
                new_dic[answer] = [color]
            else:
                new_dic[answer].append(color)
        gen.append(new_dic)
        moves = []
        for i in range(k):
            total = 0
            for color in new_dic[list(new_dic.keys())[i]]:
                total += 1
            moves.append(total - old_lens[i])
        current_gen += 1
    #  print("Differences in gen " + str(curr_gen) + " : " + str(moves))
        for i in range(k):
            if moves[i] != 0:
                break
            if i == k - 1:
                count += 1
    count = 0
    final_dic = {}
    for key in new_dic:
        if len(new_dic[key]) == 1:
            final_dic[new_dic[key][0]] = "Goalkeeper/Referee"
            count = count - 1
        else:
            for color in new_dic[key]:
                final_dic[color] = count
        count += 1
    return final_dic


                
    


