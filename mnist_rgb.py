from argparse import ZERO_OR_MORE
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'
PATH = "./model.pt"

learning_rate = 1e-3
batch_size = 1

mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
    #transforms.Normalize((0.5,),(0.5,))
])

train_data_mnist = datasets.MNIST('.\datasets', train = True, download = True, transform = mnist_transform)
print(train_data_mnist.data.shape)
train_loader = torch.utils.data.DataLoader(train_data_mnist, batch_size = batch_size, shuffle = True)

def print_image(img):
    plt.imshow(img)
    plt.show()

def rotate_img(img, b, scale = 30):
    
    a = np.random.uniform(low = -1*scale, high = scale, size = 1)
    a = a * math.pi / 180

    height, width, _ = img.shape
    rotated_image = np.zeros((28, 28, 3))
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
                rotated_image[r][c][:] = img[y][x][:]
            else:
                rotated_image[r][c][:] = b
    rotated_image = rotated_image.astype(np.uint8)
    return rotated_image

def zoom_img(img, b, offset_scale = 3, zoom_scale = 20):
    zoom_x = np.random.uniform(low = 100-zoom_scale, high = 133)/100
    zoom_y = np.random.uniform(low = 100-zoom_scale, high = 133)/100
    x_offset, y_offset = np.random.randint(-1*offset_scale, offset_scale), np.random.randint(-1*offset_scale, offset_scale)
 
    height, width, _ = img.shape
    zoomed_image = np.zeros((28, 28, 3))
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
                zoomed_image[r][c][:] = img[y][x][:]
            else:
                zoomed_image[r][c][:] = b
    zoomed_image = zoomed_image.astype(np.uint8)
    return zoomed_image

def make_pretty_background():
    background = []
    for i in range(3):
        background.append(np.random.uniform(low = 0, high = 255))

    background_rgb = np.zeros((28, 28, 3))
    a, b = np.random.random(size=2)-0.5
    func_list = [math.sin, math.cos, math.tanh]
    func = np.random.uniform(low = 0, high = 3, size = 3).astype(np.uint8)
    for r in range(28):
        for c in range(28):
            mult = (np.array([func_list[func[0]](a*r + b*c), func_list[func[1]](a*r + b*c), func_list[func[2]](a*r + b*c)]) + 1)/2
            current_back = np.clip(background + 255*mult, 0, 255)
            background_rgb[r][c] = current_back
    
    #background_rgb = background_rgb.astype(np.uint8)
    return background_rgb

def make_noisy_background(rgb = True):

    noise_lv = np.random.uniform(low = 5, high = 10)
    if rgb:
        background_rgb = np.zeros((28, 28, 3))
    elif not rgb:
        background = np.random.uniform(low = 0, high = 255)
        background_rgb = np.zeros((28, 28))

    back_offset = np.random.uniform(low = 60, high = 190, size = 3)
    for r in range(28):
        for c in range(28):
            mult = (np.random.random(size = 3 if rgb else 1)-0.5)/noise_lv
            current_back = np.clip(back_offset + 255*mult, 0, 255)
            background_rgb[r][c] = current_back
    
    background_rgb = background_rgb.astype(np.uint8)
    return background_rgb

def make_fig_background(n = 1):
    complicated = np.random.randint(low = 1, high = 3)
    if complicated == 1:
        background = [255, 255, 255]
    else:
        background = [57, 57, 57]
    fig_arr= np.zeros((28,28,3))
    for i in range(n):
        fig = np.random.randint(low = 1, high = 20)
        fig = Image.open("./aug_figure/" + str(fig) + ".png")
        
        fig_arr += np.array(fig)
        
        fig_arr = rotate_img(fig_arr, background, 180/n)
        fig_arr = zoom_img(fig_arr, background, 7, 30)
    fig_arr = np.clip(fig_arr, 0, 255)/3
    return fig_arr
    
def make_rgb_background():
    background = np.zeros((28,28,3))
    r, g, b = np.random.randint(low = 0, high = 256, size = 3)
    for x in range(28):
        for y in range(28):
            background[x][y] = [r, g, b]
    #background = background.astype(np.uint8)
    return background

def make_rgb(img, bopt, fig, soft):

    background_opt = [make_rgb_background, make_noisy_background, make_pretty_background]
    
    #rotate first
    current = np.zeros((28, 28, 3))
    for i in range(3):
        n = np.where(img != 0, img*255, 0)
        current[:,:, i] = n
    current = rotate_img(current, [0,0,0])
    current = zoom_img(current, [0,0,0])
    #add background
    indicate_num = np.random.randint(low = 0, high = 50)
    background_rgb = background_opt[bopt]()
    a = np.random.random(size=1)
    while 0.3 < a < 0.7:
        a = np.random.random(size=1)

    if fig:
        fig_num = np.random.randint(low = 1, high = 3)
        background_fig = make_fig_background(fig_num)

        if soft:
            background_rgb = np.clip((background_rgb  - (255-background_fig)/4),0,255).astype(np.uint8)
            
            for i in range(3):
                n = np.where(current[:,:,i] > indicate_num, current[:,:,i]*a, background_rgb[:,:,i])
                current[:,:, i] = n
        else:
            for i in range(3):
                n = np.where(current[:,:,i] > indicate_num, current[:,:,i]*a, background_rgb[:,:,i])
                current[:,:, i] = n
            current = np.clip((current - (255-background_fig)/4),0,255).astype(np.uint8)
    else:
        for i in range(3):
            n = np.where(current[:,:,i] > indicate_num, current[:,:,i]*a, background_rgb[:,:,i])
            current[:,:, i] = n
        current = np.clip(current,0,255).astype(np.uint8)

    return current


def rgb_to_gray(rgb):
    return np.dot(rgb,[0.299, 0.587, 0.114]).astype(np.uint8)


def write_augset_png(X, n):
    path = "./MNIST_RGB/"
    for i in range(3):
        Image.fromarray(make_rgb(X, i, True, True)).save(path + str(n+i*3+0) + ".png")
        Image.fromarray(make_rgb(X, i, True, False)).save(path + str(n+i*3+1) + ".png")
        Image.fromarray(make_rgb(X, i, False, False)).save(path + str(n+i*3+2) + ".png")

def write_augset_csv(X, Y):
    result_X = []
    result_Y = []
    for i in range(3):
        result_X.extend([rgb_to_gray(make_rgb(X, i, True, True)),rgb_to_gray(make_rgb(X, i, False, False))])
        result_Y.extend([Y.item(), Y.item()])
    return result_X, result_Y


train_X = []
train_Y = []
#fig = plt.figure()
for i, (X, Y) in enumerate(train_loader):
    #write_augset_png(X, Y, i*9+1)
    tmp_x, tmp_y = write_augset_csv(X, Y)
    train_X.extend(tmp_x)
    train_Y.extend(tmp_y)
    
    # for i in range(6):
    #     ax = plt.subplot(1,6,i+1)
    #     ax.imshow(train_X[i], cmap = 'gray', vmin = 0, vmax = 255)
    
    # plt.show()
    if i % 100 == 0:
        print("processing : ", str(i/len(train_loader)*100) + "%")
    
#np.save("train_data", train_X)
#np.save("train_label", train_Y)