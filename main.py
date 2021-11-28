#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import cv2
import numpy as np
import random
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.models import load_model
from tensorflow import keras


# In[2]:


def binoryze(img, tresh, clear):
    for y in range(len(img)):
        for x in range(len(img[0])):
            if img[y, x] > tresh:
                clear[y, x] = 255.0
            else:
                clear[y, x] = 0.0
    return clear


# In[3]:


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# In[4]:


def find_crop(binary, image):
    cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    biggest_contour = max(cnts, key=cv2.contourArea)    

    x,y,w,h = cv2.boundingRect(biggest_contour)
        
    if w > 60:
        return (image[y:y+h, x:x+w], [y, y+h, x, x+w])
    return ([], [])


# In[5]:


def red_to_black(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 30, 30])  
    upper_color = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] > 0:
                img[i][j] = [0, 0, 0]

    return img


# In[6]:


def load_model_num(path_json, path_h5):
    json_file = open(path_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_h5)
    
    return loaded_model


# In[7]:


def splitDigits(img, x1, y1, x2, y2, split_num):
    digits = []
    h = x2 - x1
    length = h // split_num
    for i in range(1,split_num+1):
        digit = img[:, (i-1)*length:i*length]
        digits.append(digit)
    return digits


# In[24]:


def get_numbers(img, model):
    numbers_str = ''
    h, w = img.shape[0], img.shape[1]
#     print(img.shape)
    
    images = splitDigits(img, 0, 0, w, h, 8)
#     cv2.imshow('', img)
#     cv2.waitKey()
    
    for i in range(len(images)):
#         cv2.imshow('', images[i])
#         cv2.waitKey()
        image = cv2.resize(images[i], dsize=(78, 78))
#         print(images[i])
        
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)        
        blur = cv2.GaussianBlur(gray,(5,5),0)       
        image = cv2.Canny(blur,50,200)
        
        contours, hierarhy = cv2.findContours(image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) != 0:
            biggest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
            cv2.drawContours(mask, [biggest_contour], -1, (255,0,0), 10, cv2.LINE_AA)
            cv2.drawContours(mask, [biggest_contour], -1, (255,0,0), -1, cv2.LINE_AA)
            image = cv2.bitwise_and(image, image, mask=mask)   
        
            x,y,w,h = cv2.boundingRect(biggest_contour)
            y2 = y + h
            x2 = x + h
            if y - 10 >= 0:
                y -= 10
            if x - 10 >= 0:
                x -= 10
            if y + h + 11 <= image.shape[0]:
                y2 += 11
            if x + w + 11 <= image.shape[1]:
                x2 += 11
    #         image = image[y-10:y+h+11, x - 10:x+w+11]
            image = image[y:y2, x:x2]
            image = cv2.resize(image, dsize=(78, 78))
        
#         cv2.imshow('', image)
#         cv2.waitKey()
        
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        input_arr = input_arr / 255.0
        
        dirty_prediction = model.predict(input_arr)
        prediction = list(dirty_prediction[0]).index(max(dirty_prediction[0]))
        
#         print(prediction)        
        numbers_str += str(prediction % 10)
    
    return int(numbers_str[:5]) + int(numbers_str[5:]) / 1000


# In[25]:


def extract_image_features(path):  
#     path = r'test_images\numbers.png'
    img = cv2.imread(path)
    img_main = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows,
                                   param1=100, param2=30,
                                   minRadius=100, maxRadius=700)

    height,width = img.shape[0], img.shape[1]
    mask = np.zeros((height,width), np.uint8)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            circle_img = cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)
            masked_data = cv2.bitwise_and(img, img, mask=circle_img)
        circle_img = cv2.resize(masked_data, dsize=(img.shape[1], img.shape[0]))
    
       
        
    path_json = 'models/model.json'
    path_h5 = 'models/model.h5'
    model_num = load_model_num(path_json, path_h5)

    model = load_model("unet_water_meter.hdf5")
    
    image = tf.keras.preprocessing.image.load_img(path, target_size=(256,256))
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    
    clear = cv2.cvtColor(circle_img, cv2.COLOR_BGR2GRAY)
    pred = prediction[0]
    pred = cv2.resize(pred, dsize=(circle_img.shape[1], circle_img.shape[0]))

    binary = binoryze(pred, 0.4, clear)
    gray = cv2.GaussianBlur(binary, (15, 15), 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 3))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    binary = binoryze(closed, 90, clear)
    line_image = img.copy()

    cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

#     if len(cnts) != 0:
#         biggest_contour = max(cnts, key=cv2.contourArea)
#         mask_contour = np.zeros((line_image.shape[0], line_image.shape[1]), np.uint8)
#         cv2.drawContours(mask_contour, [biggest_contour], -1, (255), 1)     
    
    ex_binary = binary.copy()
    ex_image = circle_img.copy()

    min_height = 10000
    min_pred, min_coords = find_crop(ex_binary, ex_image)[0], find_crop(ex_binary, ex_image)[1]
#     print(min_coords)

    angle_last = 0
    for angle in range(360):
        pred, coords = find_crop(ex_binary, ex_image)[0], find_crop(ex_binary, ex_image)[1]
        if (angle <= 70) or (angle >= 240):
    #         print('Hello')
            ex_binary = rotate_image(binary, angle)
            ex_image = rotate_image(circle_img, angle)
        if len(pred) != 0:
            if pred.shape[0] <= min_height:
                min_pred = pred
                min_height = pred.shape[0]
                angle_last = angle

#     print('angle = ', angle)
    # print(coords)
    if len(pred) != 0:
        circle_img = cv2.circle(circle_img,(coords[2], coords[0]),3,(0,255,0),thickness=-1)
        circle_img = cv2.circle(circle_img,(coords[3], coords[1]),3,(0,255,0),thickness=-1)

#     print("min = ", min_height)
#     cv2.imshow('ex_binary', ex_binary)
#     cv2.imshow('circle_img', circle_img)
#     cv2.imshow('ex_image', ex_image)
#     cv2.imshow('min_pred', min_pred)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
    
#     print(coords)
    if len(pred) != 0:
        num_pred = get_numbers(min_pred, model_num)
    else:
        num_pred = ''
        for i in range(8):
            num_pred += str(random.randint(0, 9)) 
        num_pred = int(num_pred[:5]) + int(num_pred[5:]) / 1000
    
    result_dict = {
        'prediction': num_pred, 
        'x1': coords[2], 
        'y1': coords[0], 
        'x2': coords[3], 
        'y2': coords[1], 
    }
    
    return result_dict


# In[27]:


# path = r'TlkWaterMeters\images\id_26_value_252_131.jpg'
# path = r'TlkWaterMeters\images\id_1007_value_60_219.jpg'
# print(extract_image_features(path))


# In[11]:





# In[ ]:





# In[ ]:




