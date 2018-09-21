#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/17 下午9:23
# @Author  : Lynn
# @Site    : 
# @File    : Camera.py
# @Software: PyCharm
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

############################################################################################
#catch_an_image: 使用摄像头捕获一张图片。
#       c_id:  0 - 本地摄像头， 1 - 外部摄像头
def catch_an_image(c_id = 0):
    cap = cv2.VideoCapture(c_id)
    ret, img = cap.read()
    cap.release()
    return img

############################################################################################
#real_images: 实时捕获摄像头。
#       c_id:  0 - 本地摄像头， 1 - 外部摄像头
def real_images(c_id = 0):
    cap = cv2.VideoCapture(c_id)

    while True:
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

############################################################################################
#测试捕获一张图片
def test_for_image():
    img = catch_an_image()
    cv2.imshow('picture', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

############################################################################################
#测试实时录像
def test_for_real_images():
    real_images()

############################################################################################
if __name__ == '__main__':
    test_for_image()
    # test_for_real_images()
