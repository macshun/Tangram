#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 下午6:34
# @Author  : Lynn
# @Site    : 
# @File    : analysis.py
# @Software: PyCharm
import numpy as np
import imutils
import cv2
from PIL import Image
import time
import math
import pattern
from tools import *

############################################################################################
# 轮廓上的所有点
cnt_point = []
# 图形的顶点
vertex = []

no_shape = {
    '0': [],
    '1': [],
    '2': [],
    '3': [],
    '4': [],
    '5': [],
    '6': []
}

############################################################################################
# 初始化 vertex, cnt_point
def initial_cntpoint_and_vertex():
    for i in range(0, 7):
        cnt_point.append([])
        # vertex对每个图形存储对应顶点编号，如vertex[0][0]=(x,y)
        # vertex[0]第0个图形，vertex[0][0]第0个图形的第0个顶点
        vertex.append([])


############################################################################################
#形状分析
class Analysis:
    def __init__(self):
        self.shapes = {'triangle': 0, 'square': 0, 'polygons': 0, 'circles': 0,'parallelogram':0}
    def draw_text_info(self, image):
        c1 = self.shapes['triangle']
        c2 = self.shapes['square']
        c3 = self.shapes['polygons']
        c4 = self.shapes['circles']
        c5=self.shapes['parallelogram']
        cv2.putText(image, "triangle: "+str(c1), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv2.putText(image, "square: " + str(c2), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv2.putText(image, "parallelogram: " + str(c5), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        return  image
    # load the image, convert it to grayscale, blur it slightly,
    # and threshold it
    def analy(self, image):
        k=0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dict = getColorList()
        num = 0
        for d in dict:
            if d != 'gray' and d != 'white' and d != 'black':
                num += 1
                if num == 1:
                    thresh = cv2.inRange(hsv, dict[d][0], dict[d][1])
                if len(dict[d])==4:
                    mask1=cv2.inRange(hsv, dict[d][0], dict[d][1])
                    mask2=cv2.inRange(hsv, dict[d][2], dict[d][3])
                    mask = cv2.addWeighted(mask1, 1, mask2, 1, 0)
                else:
                    mask = cv2.inRange(hsv, dict[d][0], dict[d][1])
                #cv2.imshow(d,mask)
                #cv2.waitKey(0)
                thresh = cv2.addWeighted(thresh, 1, mask, 1, 0)
        blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=4)
        thresh = cv2.dilate(closed, None, iterations=4)
        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        #print(len(cnts))
        # loop over the contours
        for c in cnts:
            #print(c)
            # 轮廓逼近
            epsilon = 0.05 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            # 分析几何形状
            corners = len(approx)
            shape_type = ""
            # 图形顶点
            for i in range(0, len(approx)):
                p = Point()
                p.x = approx[i][0][0]
                p.y = approx[i][0][1]
                vertex[k].append(p)
                cv2.circle(image, (approx[i][0][0], approx[i][0][1]), 2, (255, 0, 0), 2)
            if corners == 3:
                count = self.shapes['triangle']
                count = count + 1
                self.shapes['triangle'] = count
                shape_type = "triangle"
                # 编号
                # ul为三角形的直角顶点
                u1 = max_distance(vertex[k][0], vertex[k][1], vertex[k][2])
                # 前两个存斜边顶点，第三个存直角顶点,直角点存y小的
                if u1 != vertex[k][2]:
                    if vertex[k][0] == u1:
                        vertex[k][0] = vertex[k][2]
                    else:
                        vertex[k][1] = vertex[k][2]
                    vertex[k][2] = u1
                if (vertex[k][0].y > vertex[k][1].y):
                    t = vertex[k][1]
                    vertex[k][1] = vertex[k][0]
                    vertex[k][0] = t
            if corners == 4:
                # 用红色表示有旋转角度的矩形框架
                rect = cv2.minAreaRect(approx)
                w=rect[1][0]
                h=rect[1][1]
                ar=w/float(h)
                print(ar)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
                if ar>=0.70 and ar<=2.00:
                    count = self.shapes['square']
                    count = count + 1
                    self.shapes['square'] = count
                    shape_type = "square"
                    # 正方形编号
                    min = Xmin(vertex[k])
                else:
                    count=self.shapes['parallelogram']
                    count=count+1
                    self.shapes['parallelogram']=count
                    shape_type="parallelogram"
                    """
                    for i in range(0,4):
                        print(str(vertex[k][i].x) + "," + str(vertex[k][i].y))
                        print(str(box[i][0]) + "," + str(box[i][1]))
                        """
                    # 对平行四边形进行编号（与box重合的点为0）
                    break_flag = False
                    while break_flag == False:
                        for i in range(0, 4):
                            while break_flag == False:
                                for j in range(0, 4):
                                    if abs(vertex[k][i].x - box[j][0]) <= 10 and abs(
                                            vertex[k][i].y - box[j][1]) <= 10:
                                        min = i
                                        #print(min)
                                        break_flag = True
                if min != 0:
                    Reverse(vertex[k], min, 3)
                    Reverse(vertex[k], 0, min - 1)
                    Reverse(vertex[k], 0, 3)
            if corners >= 10:
                count = self.shapes['circles']
                count = count + 1
                self.shapes['circles'] = count
                shape_type = "circles"
            if 4 < corners < 10:
                count = self.shapes['polygons']
                count = count + 1
                self.shapes['polygons'] = count
                shape_type = "polygons"
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            #根据螺丝的半径，圆心重新确定识别的点（cx-R,cy-R)
            color=Color(cX-5,cY-5,im)
            #color=Color(vertex[k][0].x,vertex[k][0].y,im)
            # draw the contour and center of the shape on the image
            cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
            cv2.putText(image, str(color)+" "+shape_type, (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(image, (cX,cY), 2, (255, 0, 0), 2)
            no_shape[str(k)].append(shape_type)
            no_shape[str(k)].append(color)
            no_shape[str(k)].append((cX, cY))
            for i in range(0,len(vertex[k])):
                cv2.putText(image,str(i),(vertex[k][i].x,vertex[k][i].y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            k+=1
        cv2.imshow("Analysis Result", self.draw_text_info(image))
        cv2.imwrite("image/pattern.jpg",image)
        cv2.waitKey(0)