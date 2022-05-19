
import h5py
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import cv2
import math
import os
import re

def make_mnist(img, background = np.zeros((28,28)), reverse = False):

    if len(img.shape)==3:
        height, width, _ = img.shape
        new_mnist = background
    elif len(img.shape)==2:
        height, width = img.shape
        new_mnist = background

    mid_row = int((height + 1) / 2)
    mid_col = int((width + 1) / 2)
    for r in range(height):#0, 28
        for c in range(width):# 0, 10
            y = (14 - mid_row) + r
            x = (14 - mid_col) + c
            if not reverse:
                if img[r][c] <= 200:
                    new_mnist[y][x] = img[r][c]
                    
            else:
                if img[r][c] > 200:
                    new_mnist[y][x] = img[r][c]
    return new_mnist

def rotate_img(img, b = 0, scale = 30):
    
    a = np.random.uniform(low = -1*scale, high = scale, size = 1)
    a = a * math.pi / 180
    if len(img.shape)==3:
        height, width, _ = img.shape
        rotated_image = np.zeros((28, 28, 3))
    elif len(img.shape)==2:
        height, width = img.shape
        rotated_image = np.zeros((28, 28))
    mid_row = int((height + 1) / 2)
    mid_col = int((width + 1) / 2)
    for r in range(height):
        for c in range(width):
            y = (r-mid_col)*math.cos(a) + (c-mid_row)*math.sin(a)
            x = -(r-mid_col)*math.sin(a) + (c-mid_row)*math.cos(a)

            y += mid_col
            x += mid_row

            x = round(x)
            y = round(y)

            if (x >= 0 and y >= 0 and x < width and y < height):
                if len(img.shape)==3:
                    
                    rotated_image[r][c][:] = img[y][x][:]
                elif len(img.shape)==2:
                    rotated_image[r][c] = img[y][x]
            else:
                if len(img.shape)==3:
                    rotated_image[r][c][:] = [b,b,b]
                elif len(img.shape)==2:
                    rotated_image[r][c] = b
    rotated_image = rotated_image.astype(np.uint8)
    return rotated_image

def zoom_move_img(img, b = 0, offset_scale = 3, zoom_scale = 20):
    zoom_x = np.random.uniform(low = 100-zoom_scale, high = 100)/100
    zoom_y = np.random.uniform(low = 105, high = 105+zoom_scale)/100
    x_offset, y_offset = np.random.randint(-1*offset_scale, offset_scale), np.random.randint(-1*offset_scale, offset_scale)
    if len(img.shape)==3:
        height, width, _ = img.shape
        zoomed_image = np.zeros((28, 28, 3))
    elif len(img.shape)==2:
        height, width = img.shape
        zoomed_image = np.zeros((28, 28))
    
    mid_row = int((height + 1) / 2)
    mid_col = int((width + 1) / 2)

    for r in range(height):
        for c in range(width):
            y = (r-mid_col)*zoom_y+ y_offset
            x = (c-mid_row)*zoom_x+ x_offset

            y += mid_col 
            x += mid_row 

            x = round(x)
            y = round(y)
            if (x >= 0 and y >= 0 and x < width and y < height):
                if len(img.shape)==3:
                    
                    zoomed_image[r][c][:] = img[y][x][:]
                elif len(img.shape)==2:
                    zoomed_image[r][c] = img[y][x]
            else:
                if len(img.shape)==3:
                    zoomed_image[r][c][:] = [b,b,b]
                elif len(img.shape)==2:
                    zoomed_image[r][c] = b
    zoomed_image = zoomed_image.astype(np.uint8)
    return zoomed_image


def rgb_to_gray(rgb):
    return np.dot(rgb,[0.299, 0.587, 0.114]).astype(np.uint8)

def resize(img):
    s = max(img.shape[0], img.shape[1])
    scale_rate = 28/s
    result = cv2.resize(img, dsize = (int(img.shape[1]*scale_rate),int(img.shape[0]*scale_rate)))
    return result

def hist_equal(img):
    if len(img.shape)==3:
        for i in range(3):
            img[:,:,i] = cv2.equalizeHist(img[:,:,i])
    elif len(img.shape)==2:
        img = cv2.equalizeHist(img)
    return img

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    cl1 = clahe.apply(img)
    return cl1

def make_pretty_background(rgb = True):
    
    if rgb:
        background = []
        for i in range(3):
            background.append(np.random.uniform(low = 0, high = 255))
        background_rgb = np.zeros((28, 28, 3))
    elif not rgb:
        background = np.random.uniform(low = 0, high = 255)
        background_rgb = np.zeros((28, 28))
    
    a, b = np.random.random(size=2)-0.5
    func_list = [math.sin, math.cos, math.tanh]
    func = np.random.uniform(low = 0, high = 3, size = 3).astype(np.uint8)
    for r in range(28):
        for c in range(28):
            if rgb:
                mult = (np.array([(func_list[func[0]](a*r + b*c) + func_list[func[1]](2*a*r + b*c/2))/2, (func_list[func[1]](a*r + b*c) + func_list[func[2]](2*a*r + b*c/2))/2, (func_list[func[2]](a*r + b*c) + func_list[func[0]](a*r/2 + b*c*2)/2)]))
            else:
                mult = (func_list[func[0]](2*a*r + b*c/2)+ func_list[func[1]](a*r/2 + b*c*2)+ func_list[func[2]](a*r + b*c))/3
            current_back = np.clip(background + 30*mult, 0, 255)
            background_rgb[r][c] = current_back
    background_rgb = background_rgb.astype(np.uint8)
    return background_rgb

def make_noisy_background(rgb = True):

    noise_lv = np.random.uniform(low = 5, high = 10)
    if rgb:
        background_rgb = np.zeros((28, 28, 3))
    elif not rgb:
        background = np.random.uniform(low = 0, high = 255)
        background_rgb = np.zeros((28, 28))

    back_offset = np.random.uniform(low = 30, high = 220, size = 3)
    for r in range(28):
        for c in range(28):
            mult = (np.random.random(size = 3 if rgb else 1)-0.5)/noise_lv
            current_back = np.clip(back_offset + 255*mult, 0, 255)
            if rgb:
                background_rgb[r][c] = current_back
            else:
                background_rgb[r][c] = current_back[0]
    
    background_rgb = background_rgb.astype(np.uint8)
    return background_rgb

def make_background():
    background = np.zeros((28,28))
    b = np.random.randint(low = 0, high = 256)
    for x in range(28):
        for y in range(28):
            background[x][y] = b
    background = background.astype(np.uint8)
    return background

path = './etc/'
file_list = os.listdir(path)
file_list = [file for file in file_list if file.endswith(".png")]
file_list.sort(key=lambda f: int(re.sub('\D', '', f)))
print(file_list)
label = open("./etc/labels.txt", "r")
labels = label.readlines()
for i in range(len(labels)):
    labels[i] = int(labels[i].strip())
print(labels)
current = 0
cnt = 0
train_X = []
train_Y = []
for file in file_list:
    try:
        png = pil.open(path + file)
    except:
        continue
    full_png = np.array(png)
    if full_png.shape[1] < 30:
        continue
    
    label = labels[current]
    
    current_num = full_png
    current_num = rgb_to_gray(current_num)
    
    
    rev_flag = True
    current_num = resize(current_num)
    for i in range(50):
        if i % 3 == 0:
            rev_flag = True
        else:
            rev_flag = False
        pbg1 = make_noisy_background(False)
        pbg2 = make_pretty_background(False)
        pbg3 = make_background()
        noisy = make_mnist(current_num, pbg1, reverse = rev_flag)
        pretty = make_mnist(current_num, pbg2, reverse = rev_flag)
        origin = make_mnist(current_num, pbg3, reverse = rev_flag)

        noisy = zoom_move_img(noisy)
        noisy = rotate_img(noisy)
        
        pretty = zoom_move_img(pretty)
        pretty = rotate_img(pretty)
        
        origin = zoom_move_img(origin)
        origin = rotate_img(origin)
        train_X.extend([noisy, pretty, origin])
        train_Y.extend([label, label, label])
        cnt += 3
    current += 1
    
print(cnt)
np.save("train_data_etc", train_X)
np.save("train_label_etc", train_Y)