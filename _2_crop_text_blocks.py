import cv2
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

import matplotlib.pyplot as plt

from minisom import MiniSom

# Code
def crop_text_blocks():
    img = cv2.imread("result/res_img_to_translate.jpg")
    file = open("result/res_img_to_translate.txt", "r")

    metric_features = ['ltx','lty','rtx','rty','tdx','rdy','ldx','ldy','cx','cy']
    df = pd.DataFrame(columns=metric_features)

    data_to_return = {}

    counter = 1
    while True:
        content=file.readline()
        if not content:
            break
        
        if len(content) > 1:
            print(content)
            coordinates = content.strip().split(',')
            coordinates = [int(x) for x in coordinates]
            coordinates.append(coordinates[0] + (coordinates[2] - coordinates[0]) / 2)
            coordinates.append(coordinates[1] + (coordinates[7] - coordinates[1]) / 2)
            print(coordinates)
            
            df.loc[-1] = coordinates
            df.index = df.index + 1
            df = df.sort_index()
            # 21,60,36,60,36,77,21,77
            #crop_img = img[60:77, 21:36]
            crop_img = img[int(coordinates[1]):int(coordinates[7]), int(coordinates[0]):int(coordinates[2])]
            #filename = 'result/blocks/block_' + str(counter) + '.jpg'
            
            filename = 'block_' + str(coordinates[1]) + "_" + str(coordinates[7]) + "_" + str(coordinates[0]) + "_" + str(coordinates[2])
            data_to_return[filename] = coordinates
            path = 'result/blocks/' + filename + '.jpg'
            
            counter += 1
            cv2.imwrite(path, crop_img)
            #cv2.imshow("cropped", crop_img)
            #cv2.waitKey(0)
            #if counter == 2:
            #    break

    file.close()
    #cv2.waitKey(0)
    
    print(df)
    
    return data_to_return
    