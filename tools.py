#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 下午6:17
# @Author  : Lynn
# @Site    : 
# @File    : tools.py
# @Software: PyCharm

import cv2
from PIL import Image
import numpy as np
import collections
import colorsys
import math
from PIL import Image
from PIL import ImageDraw

############################################################################################
# 生成深蓝色绘图画布
def generate_darkblue_canvas():
    array = np.ndarray((1190, 1190, 3), np.uint8)
    array[:, :, 0] = 0
    array[:, :, 1] = 0
    array[:, :, 2] = 100
    image = Image.fromarray(array)
    # 创建绘制对象
    draw = ImageDraw.Draw(image)
    return (image,draw)

############################################################################################
#Point:描述点坐标
class Point():
    x = 0
    y = 0

############################################################################################
#rgb转hsv
def rgb_hsv(x,y,image):
    color = image.getpixel((x, y))
    new_color = colorsys.rgb_to_hsv(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    hsv = [int(new_color[0] * 360 / 2), int(new_color[1] * 255), int(new_color[2] * 255)]
    return  hsv

############################################################################################
#筛查颜色
def Color(x,y,image):
    new_colors=rgb_hsv(x,y,image)
    #print(new_colors)
    dict=getColorList()
    for i in dict:
        if dict[i][0][0] <= new_colors[0] <= dict[i][1][0] and dict[i][0][1] <= \
                new_colors[1] <= dict[i][1][1] and dict[i][0][2] <= new_colors[2] <= \
                dict[i][1][2]:
            return i

############################################################################################
#contrast_brightness_image:亮度、对比度调整
def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    #dst = src1*alpha + src2*beta + gamma
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如下
    #cv2.imshow("con-bri-demo", dst)
    #cv2.waitKey(0)
    return  dst

############################################################################################
#slope:
def slope(p1, p2,flag):
    if 0 <= abs(p1.x - p2.x) <= 15 and p1.y != p2.y:
        slope = float("inf")
    else:
        # 邻边斜率
        slope = ((-1) * (p1.y) + p2.y) / (p1.x - p2.x)
        if flag==1:
            if (abs(slope) < 0.2):
                slope = 0
            if abs(abs(slope) - 1) < 0.2:
                if slope < 0:
                    slope = -1
                else:
                    slope = 1
    return slope

############################################################################################
#某一点逆时针旋转theta（弧度）
def rotatePoint(p1,p,theta):
    p_r=Point()
    p_r.x=p.x+(p1.x-p.x)*math.cos(theta)-(p1.y-p.y)*math.sin(theta)
    p_r.y=p.y+(p1.x-p.x)*math.sin(theta)+(p1.y-p.y)*math.cos(theta)
    return p_r

############################################################################################
# 数组倒序
def Reverse(vertex, k, m):
    # N=len(vertex)
    j = m
    for i in range(k, k + int((m - k + 1) / 2)):
        u = vertex[j]
        vertex[j] = vertex[i]
        vertex[i] = u
        j -= 1

############################################################################################
# 寻找正方形x最小的点的下标
def Xmin(vertex):
    l = len(vertex)
    min = 0
    for i in range(0, l):
        if vertex[i].x < vertex[min].x:
            min = i
    return min

############################################################################################
# 求三角形直角边的顶点作为第一个顶点
def max_distance(p1, p2, p3):
    d1 = pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)
    d2 = pow(p2.x - p3.x, 2) + pow(p2.y - p3.y, 2)
    d3 = pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2)
    if abs(d2 - d3) < abs(d1 - d3) and abs(d2 - d3) < abs(d2 - d1):
        return p3
    elif abs(d1 - d3) < abs(d1 - d2) and abs(d1 - d3) < abs(d2 - d3):
        return p1
    else:
        return p2

############################################################################################
#由斜率求角度
def angle(p1,p2,flag):
    temp1 = slope(p1, p2,flag)
    temp1 = math.atan(temp1)
    angle = math.degrees(temp1)
    angle=round(angle)
    if angle<0:
        angle=180+angle
    return angle

############################################################################################
#沿任意一条直线翻转
def Translation(p,a,k,theta):
    p_r = Point()
    temp=Point()
    if k!=float('inf') and k!=0:
        p_r.x = p.x
        p_r.y = p.y - a
        if k!=0:
            temp.x = p.x * math.cos(theta) - (p.y-a) * math.sin(theta)
            temp.y = p.x * math.sin(theta) + (p.y-a) * math.cos(theta)
        p_r.y=p_r.y*-1
        temp.y=temp.y*-1
        if k!=0:
            p_r.x = temp.x * math.cos(theta) + temp.y * math.sin(theta)
            p_r.y = -temp.x * math.sin(theta) + temp.y * math.cos(theta)
        p_r.y=p_r.y+a
    elif k==float('inf'):
        p_r.x=p.x-a
        p_r.y=p.y
        p_r.x=-1*p_r.x+a
    else:
        p_r.y=p.y-a
        p_r.x=p.x
        p_r.y=-1*p_r.y+a
    return p_r

############################################################################################
def draw_test(p1,p2,p3,draw,color):
    # 绘制多边形
    draw.polygon((p1.x,p1.y,p2.x,p2.y,p3.x,p3.y), color)
    return

############################################################################################
#单个图形的旋转
def Rotate(mid,j,ang_gap,num,vertex):
    if ang_gap < 0:
        ang_gap = abs(ang_gap)
        ang_gap = math.radians(ang_gap)
        ang_gap = -1 * ang_gap
    # 顺时针旋转
    elif ang_gap > 0:
        ang_gap = math.radians(ang_gap)
    for u in range(0, num):
        vertex[j][u] = rotatePoint(vertex[j][u], mid, ang_gap)

############################################################################################
#getColorList:生成hsv颜色字典,Hue-色调、Saturation-饱和度、Value-值s
def getColorList():
    dict = collections.defaultdict(list)

    #黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list

    # #灰色

    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['gray']=color_list

     #白色
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list

    #粉色
    lower_pink=np.array([166,86,178])
    upper_pink=np.array([180,165,255])
    color_list=[]
    color_list.append(lower_pink)
    color_list.append(upper_pink)
    dict['pink']=color_list

    # 红色

    lower_red = np.array([166, 166, 80])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red'] = color_list

    # 红色2
    lower_red = np.array([0, 140, 130])
    upper_red = np.array([5, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list

    #橙色
    lower_orange = np.array([6, 43, 46])
    upper_orange = np.array([24, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    #黄色
    lower_yellow = np.array([25, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    #绿色

    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    #青色

    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    #蓝色

    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色

    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([165, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list

    return dict

############################################################################################

