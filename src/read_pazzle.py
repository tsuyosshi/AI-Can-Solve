from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import tensorflow as tf
import keras
import math
from util import load_img

def fill_unnecessary_area(img, cntrs, idx, back_color=0):
    for i, c in enumerate(cntrs):
        if i == idx:
            continue
        x, y, w, h = cv2.boundingRect(c)
        img[y:(y + h), x:(x + w)] = back_color
    return img

def append_contour_to_df(df, cntr, label_no, cntr_no, path):
    x, y, w, h = cv2.boundingRect(cntr)
    df = pd.concat([df, pd.DataFrame({
        'label_no': [label_no],
        'cntr_no': [cntr_no], 
        'x': [x], 
        'y': [y], 
        'w': [w], 
        'h': [h], 
        'path': [path]})]).reset_index(drop=True)
    return df

def detect_empty_block(df):
    x_mi = df['x'].min()
    x_ma = df['x'].max()
    y_mi = df['y'].min()
    y_ma = df['y'].max()
    W = x_ma - x_mi
    H = y_ma - y_mi
    sz = df.shape[0] + 1 # 一つ空白マスなので検出した数字の数+1が全体のマスの数
    w = W / sz
    h = H / sz

    x = x_mi
    y = y_mi

    for i in range(df.shape[0]):
        pos_idx = nearest_pos(df, x, y)
        print(pos_idx)
        xi = df.loc[pos_idx, 'x']
        yi = df.loc[pos_idx, 'y']

        # pos_idx が明らかにおかしい場合
        if abs(x-xi) > w/2 or abs(y-yi) > h/2:
            return pos_idx, x, y

        if i % math.sqrt(sz) == 0:
            x = x_mi
            y = y + h
        else:
            x = x + w
            y = y_mi

    return -1, x, y


def nearest_pos(df, x, y):
    idx = -1
    dist = 10000000000

    for i in range(df.shape[0]):
        xi = df.at[i, 'x']
        yi = df.at[i, 'y']
        d = (x - xi)*(x - xi) + (y - yi)*(y - yi)
        if dist > d:
            dist = d
            idx = i

    return idx

def read_image(image):

    gray_image = image.copy()

    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    ret, image_thresh = cv2.threshold(blur_image, 80, 255, cv2.THRESH_BINARY)

    # 連結成分のラベリングを行う。
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_thresh)


    stats = sorted(stats, key=lambda x:x[4], reverse=True)

    contours_df = pd.DataFrame(columns=['label_no', 'cntr_no', 'x', 'y', 'w', 'h', 'path'])
    contour_nums = [0] * n_labels

    if os.path.exists('./image/slide/trimmed/'):
        shutil.rmtree('./image/slide/trimmed/')
    
    os.mkdir('./image/slide/trimmed/')

    cv2.imwrite('./image/slide/trimmed/img_thresh.jpg', 255-image_thresh)

    for i in range(n_labels):

        if not os.path.exists(f'./image/slide/trimmed/00{i}/'):
            os.mkdir(f'./image/slide/trimmed/00{i}/')

        x, y, width, height, area = stats[i]

        # 比率 ratio でトリミング
        ratio = 0.6
        sx = int(((1-ratio)*width) / 2)
        tx = int(((1+ratio)*width) / 2)
        sy = int(((1-ratio)*height) / 2)
        ty = int(((1+ratio)*height) / 2)
        img = image_thresh[y+sy:y+ty, x+sx:x+tx]
        img = 255 - img

        print(img.shape)

        if img.shape[0] > 1000 and img.shape[1] > 1000:
            continue

        if img.shape[0] < 50 and img.shape[1] < 50:
            continue

        cv2.imwrite(f'./image/slide/trimmed/00{i}/00{i}.jpg', img)


        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sz = 0
        thresh_holds = 500

        for j, cntrs in enumerate(contours):
            if cv2.contourArea(cntrs) < thresh_holds:
                continue

            img_tmp = fill_unnecessary_area(img.copy(), contours, j)
            image_path = f'./image/slide/trimmed/00{i}/00{i}_00{sz}.jpg'
            cv2.imwrite(image_path, img_tmp)
            contours_df = append_contour_to_df(contours_df, cntrs, i, sz, image_path)
            sz = sz + 1

        contour_nums[i] = sz

    contours_df = contours_df.set_index(['label_no', 'cntr_no'])

    # for i in range(n_labels):
    #     if contour_nums[i] < 2 or contour_nums[i] > 3:
    #         continue
    #     for j in range(contour_nums[i]):
    #         x, y, w, h, path = contours_df.loc[(i, j)]
    #         image = cv2.imread(path)

    contours_df.to_csv('./work/contours.csv')

    return contours_df, contour_nums, stats

def predict_nums(image):

    contours_df, contour_nums, stats = read_image(image) 
    numbers_df = pd.DataFrame(columns=['value', 'x', 'y'])

    n_labels = len(contour_nums)
    model = keras.models.load_model("./work/model")

    for i in range(n_labels):

        if contour_nums[i] < 1 or contour_nums[i] > 2:
            continue

        x_s, y_s, w_s, h_s, a_s = stats[i]

        num_ls = []
        for j in range(contour_nums[i]):
            x, y, w, h, path = contours_df.loc[(i, j)]
            image = load_img(path)
            image = image.reshape(1, 28, 28)
            pred = model.predict(image).argmax()
            num_ls.append((x, pred))
        # x座標でsort
        num_ls.sort(key=lambda x: x[0])
        num_ls = [v[1] for v in num_ls]
        num = int("".join(map(str, num_ls)))

        numbers_df = numbers_df.append(pd.DataFrame({
            'value': [num],
            'x': [x_s],
            'y': [y_s]
        }))
    
    numbers_df = numbers_df.reset_index(drop=True)

    # target_idx の上に空白マスを追加
    target_idx, x_emp, y_emp = detect_empty_block(numbers_df)
    df1 = numbers_df.loc[:target_idx-1, :]
    df1 = df1.append(pd.DataFrame({
        'value': [-1],
        'x': [x_emp],
        'y': [y_emp]
    }))
    df2 = numbers_df.loc[target_idx:, :]

    numbers_df = pd.concat([df1, df2]).reset_index(drop=True)


    numbers_df = numbers_df.sort_values('y').reset_index(drop=True)

    # print(numbers_df)

    size = numbers_df.shape[0]

    numbers_df['col'] = -1
    
    for i in range(size):
        numbers_df.loc[i, 'col'] = i // math.sqrt(size)

    numbers_df = numbers_df.sort_values(['col', 'x']).reset_index(drop=True)

    print(numbers_df)

    return numbers_df.shape[0], numbers_df
    
# IMAGE_PATH = './image/slide/slide_4x4_001.jpg'

# df = predict_nums(IMAGE_PATH)
# print(df)
    



    
    







