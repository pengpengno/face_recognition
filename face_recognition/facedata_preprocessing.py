
import os
import sys
import numpy as np
import cv2
import tensorflow as tf

image_size=64
file_path=r"../face_data"
face_label1="wangpeng"
face_label2="xuxin"

haar= cv2.CascadeClassifier("E:/python3.7/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
"""
this file is used for data preprocessing
main function：
1.togrey                parameter:（path）                the path is the root of image
2.resize_facedata       parameter:(path,image_size=64)    The default size of the image is 64
3.crop_face             parameter: (path)                 the path is the root of image
4.resize_image          parameter: (image)                return image after processing
"""
sess=tf.InteractiveSession()
def togrey(filepath):
    gray_image_nums=0
    invalid_imgage=0
    for file in os.listdir(filepath):
        file_fullpath=os.path.join(filepath,file)
        image=cv2.imread(file_fullpath)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # 灰度
        faces = haar.detectMultiScale(gray,1.1,3,0,(100,100))
        if len(faces)!=0:
            gray_image_nums+=1
            print("已灰度处理%d张图片" ,gray_image_nums)
            cv2.imwrite(file_fullpath,gray)
        else:
            os.remove(file_fullpath)
            invalid_imgage+=1
            print("已经删除%d张无人脸数据" ,invalid_imgage)
#调整数据为同一大小64*64
def resize_facedata(src,IMAGE_SIZE):
    faces=os.listdir(src)
    nums=0
    for face in faces:
        print(face)
        fullpath=os.path.join(src,face)
        image = cv2.imread(fullpath)
        image=cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))
        cv2.imwrite(fullpath,image)
        nums+=1
        print("{0}已完成{1}".format(face,nums))
#识别图片中的人脸，裁剪人脸区域后替换原图
def crop_face(src):
    faces=os.listdir(src)
    nums=0
    for face in faces:
        print(face)
        fullpath=os.path.join(src,face)
        image = cv2.imread(fullpath)
        faces_rect=haar.detectMultiScale(image,1.2,3)
        if len(face):
            for face_rect in faces_rect:
                x,y,w,h=face_rect
                image=np.array(image,'uint8')
                image=image[y:y+h  , x:x+w]
                cv2.imwrite(fullpath,image)
                nums+=1
                print("{0}已完成{1}".format(face,nums))
#返回单张64*64 的人脸数据，主要用在实时视频流调整待检测的人脸数据中
def resize_image(image):
    image=cv2.resize(image,(image_size,image_size))
    #cv2.imwrite("./wangpeng.jpg",image)
    return image