import h5py
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import cv2
import math

f = h5py.File('./svhn_train/digitStruct.mat', 'r')

def make_mnist(img, background = np.zeros((28,28))):

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

def svhn_img_count():
    return f["digitStruct"]["bbox"].size

def svhn_img(n):
    return f["digitStruct"]["bbox"][n][0]

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

class svhnimg:
    def __init__(self, obj):
        self.obj = obj

    def get_label_count(self):
        return np.array(f[self.obj]['label']).size

    #['height' 'label' 'left' 'top' 'width']
    def get_info(self, info):
        cnt = self.get_label_count()
        result = np.zeros(cnt, dtype=np.uint16)
        for i in range(cnt):
            label = None
            if cnt == 1:
                label = np.array(f[self.obj][info], dtype=np.uint16)
            else:
                tmp = np.array(f[self.obj][info])[i][0]
                label = np.array(f[tmp][0][0], dtype=np.uint16)
            result[i] = label
        return result

    def get_full_data(self):

        return np.array([self.get_info('left'), 
                            self.get_info('top'), 
                            self.get_info('width'), 
                            self.get_info('height')], dtype = np.uint16).T


cnt = 0
train_X = []
train_Y = []
for i in range(svhn_img_count()):
    try:
        png = pil.open("./svhn_train/" + str(i+1) + ".png")
    except:
        continue
    full_png = np.array(png)
    if full_png.shape[1] < 30:
        continue
    img = svhnimg(svhn_img(i))
    full_data = img.get_full_data()
    full_label = img.get_info('label')
    for j in range(len(full_data)):
        try:
            current_num = full_png[full_data[j][1]:full_data[j][1]+full_data[j][3],
                                    full_data[j][0]:full_data[j][0]+full_data[j][2],
                                    :]
            current_num = rgb_to_gray(current_num)
            current_label = 0 if full_label[j]>9 else full_label[j]

            pbg1 = make_noisy_background(False)
            pbg2 = make_pretty_background(False)
            pbg3 = make_background()
            
            current_num = resize(current_num)
            current_num = hist_equal(current_num)

            noisy = make_mnist(current_num, pbg1)
            pretty = make_mnist(current_num, pbg2)
            origin = make_mnist(current_num, pbg3)

            noisy = zoom_move_img(noisy)
            noisy = rotate_img(noisy)
            
            pretty = zoom_move_img(pretty)
            pretty = rotate_img(pretty)
            
            origin = zoom_move_img(origin)
            origin = rotate_img(origin)

            train_X.extend([noisy, pretty, origin])
            train_Y.extend([current_label, current_label, current_label])
            cnt += 3
            
        except:
            continue
    if i % 100 == 0:
        print("processing : ", str(i/svhn_img_count()*100) + "%")
print(cnt)
np.save("train_data_svhn", train_X)
np.save("train_label_svhn", train_Y)
f.close()