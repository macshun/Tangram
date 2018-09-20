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


def catch_an_image():
    # cap = cv2.VideoCapture(1)# 控制外部摄像头
    cap = cv2.VideoCapture(0)#控制本地摄像头
    ret, img = cap.read()
    cap.release()
    return img

def real_images():
    # cap = cv2.VideoCapture(1)#控制外部摄像头
    cap = cv2.VideoCapture(0)#控制本地摄像头

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

#测试捕获一张图片
def test_for_image():
    img = catch_an_image()
    cv2.imshow('picture', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#测试实时录像
def test_for_real_images():
    real_images()

if __name__ == '__main__':
    test_for_image()
    # test_for_real_images()
