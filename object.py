import numpy as np
import cv2
from PIL import Image
import time
import math
import pattern
from PIL import ImageDraw
import random
from tools import *
from analysis import *

# 生成深蓝色绘图画布
array = np.ndarray((1190, 1190, 3), np.uint8)
array[:, :, 0] = 0
array[:, :, 1] = 0
array[:, :, 2] = 100
image = Image.fromarray(array)
# 创建绘制对象
draw = ImageDraw.Draw(image)

initial_cntpoint_and_vertex()
print('###############test for vertex################')
print(vertex)
print(cnt_point)
print('###############test for vertex################')
# cv2.waitKey(0)
# 电子图案与实物图进行比对分析
# 1、寻找对应的三角形
def bfs(graph, v,no_shape,vertex,pattern):
    queue = []
    queue.insert(0, v)
    pattern.no_shape[str(v)][4]=True
    while (len(queue) != 0):
        top = queue[len(queue) - 1]
        pattern.no_shape[str(top)][4]=True
        mid = Point()
        color = pattern.no_shape[str(top)][1]
        an = pattern.no_shape[str(top)][3]
        if len(pattern.vertex[top]) == 3:
            for j in range(0, 7):
                if len(vertex[j]) == 3 and no_shape[str(j)][1] == color:
                    mid.x = no_shape[str(j)][2][0]
                    mid.y = no_shape[str(j)][2][1]
                    ang_gap = no_shape[(str(j))][3] - an
                    mid2 = Point()
                    mid2.x = pattern.no_shape[str(top)][2][0]
                    mid2.y = pattern.no_shape[str(top)][2][1]
                    # 先绕中心点进行旋转
                    Rotate(mid, j, ang_gap, 3, vertex)
                    # 判断是否需要翻转
                    flag = False
                    if 0 <= pattern.no_shape[str(top)][3] <= 45 or 135 <= pattern.no_shape[str(top)][3] <= 180:
                        if (vertex[j][2].y < mid.y and pattern.vertex[top][2].y < mid2.y) or (
                                vertex[j][2].y > mid.y and pattern.vertex[top][2].y > mid2.y):
                            print
                        else:
                            flag = True
                            print
                    elif 45 < pattern.no_shape[str(top)][3] <= 90 or 90 < pattern.no_shape[str(top)][3] < 135:
                        if (vertex[j][2].x < mid.x and pattern.vertex[top][2].x < mid2.x) or (
                                vertex[j][2].x > mid.x and pattern.vertex[top][2].x > mid2.x):
                            print
                        else:
                            flag = True
                            print
                    if flag == True:
                    #记录移动、翻转
                        #Rotate(mid, j, -ang_gap, 3, vertex)
                        Rotate(mid, j, 180, 3, vertex)
                        no_shape[(str(j))].append(ang_gap+180)
                    else:
                        #Rotate(mid, j, -ang_gap, 3, vertex)
                        #Rotate(mid, j, ang_gap, 3, vertex)
                        no_shape[(str(j))].append(ang_gap)
                    draw_test(vertex[j][0], vertex[j][1], vertex[j][2], draw, color)
        elif len(pattern.vertex[top]) == 4 and pattern.no_shape[str(top)][0] == 'square':
            for j in range(0, 7):
                if len(vertex[j]) == 4 and no_shape[str(j)][0] == 'square':
                    mid.x = no_shape[str(j)][2][0]
                    mid.y = no_shape[str(j)][2][1]
                    ang_gap = no_shape[(str(j))][3] - an
                    Rotate(mid, j, ang_gap, 4, vertex)
                    no_shape[str(j)].append(ang_gap)
                    draw.polygon((
                        vertex[j][0].x, vertex[j][0].y, vertex[j][1].x, vertex[j][1].y, vertex[j][2].x,
                        vertex[j][2].y,
                        vertex[j][3].x, vertex[j][3].y), color)
        elif len(pattern.vertex[top]) == 4 and pattern.no_shape[str(top)][0] == 'parallelogram':
            for j in range(0, 7):
                if len(vertex[j]) == 4 and no_shape[str(j)][0] == 'parallelogram':
                    mid.x = no_shape[str(j)][2][0]
                    mid.y = no_shape[str(j)][2][1]
                    # 根据（0，1）边角度差值进行旋转
                    ang_gap = no_shape[(str(j))][3] - an
                    Rotate(mid, j, ang_gap, 4, vertex)
                    no_shape[str(j)].append(ang_gap)
                    draw.polygon((
                        vertex[j][0].x, vertex[j][0].y, vertex[j][1].x, vertex[j][1].y, vertex[j][2].x,
                        vertex[j][2].y,
                        vertex[j][3].x, vertex[j][3].y), color)
        for i in range(0,len(graph[str(top)])):
            temp=graph[str(top)][i].no
            if pattern.no_shape[str(temp)][4]==False and temp not in queue:
                queue.insert(0,temp)
        queue.pop()


def main():
    start=time.clock()
    # image = cv2.imread("image/shoot/no_highlight/(4).jpg")
    image = cv2.imread("pictures/2.jpg")
    size=image.shape
    # img = cv2.resize(image, (int(size[1] * 0.3), int(size[0] * 0.3)), interpolation=cv2.INTER_AREA)
    img = cv2.resize(image, (int(size[1] * 0.3), int(size[0] * 0.3)), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]  # 获取图像的高和宽
    # cv2.imshow("Origin", img)  # 显示原始图像
    # cv2.waitKey(0)
    #存在高光时，适量降低图片亮度
    img = contrast_brightness_image(img, 1.2,-4)
    cv2.imshow("k",img)
    # cv2.waitKey(0)
    blured = cv2.blur(img, (5, 5))  # 进行滤波去掉噪声
    #test for bured
    cv2.imshow('blured picture', blured)
    cv2.waitKey(0)
    blured_copy=blured.copy()
    # mask：掩码图像，大小比原图多两个像素点。设输入图像大小为width * height, 则掩码的大小必须为 (width+2) * (height+2) ,
    #       mask可为输出，也可作为输入 ，由flags决定。
    mask = np.zeros((h + 2, w + 2), np.uint8)  # 掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
    # 进行泛洪填1
    cv2.floodFill(blured, mask, (w - 1, h - 1), (0, 0, 0), (50,35,50), (185, 190, 190), cv2.FLOODFILL_FIXED_RANGE)
    # test for flood fill
    # cv2.imshow('flood fill', blured)
    # cv2.waitKey(0)
    ld = Analysis()
    ld.analy(blured)
    if ld.shapes['triangle']==5 and ld.shapes['parallelogram']==1 and ld.shapes['square']==1:
        print('yes')
    else:
        print("第二遍")
        for i in range(0,7):
            #清空第一遍时加入的内容
            no_shape[str(i)]=[]
            vertex[i]=[]
        ld.shapes['triangle'] =0
        ld.shapes['parallelogram'] =0
        ld.shapes['square'] =0
        ld.analy(blured_copy)
    #实物图计算三角形斜边的角度
    print('###############test for vertex################')
    print(vertex)
    print(cnt_point)
    print('###############test for vertex################')
    for i in range(0, 7):
        angel_hypotenuse=angle(vertex[i][0], vertex[i][1],0)
        no_shape[str(i)].append(angel_hypotenuse)
    for i in range(0,7):
       pattern.no_shape[str(i)].append(False)

if __name__ == '__main__':
    start = time.clock()
    src = cv2.imread("image/mould/06.jpg")
    # src = cv2.imread("pictures/7.jpeg")

    # 当目标图案过大时进行压缩，指定大小而不是以比例压缩（否则会影响后期像素点检测）
    size = src.shape
    ld=pattern.ShapeAnalysis()
    ld.analysis(src)
    pattern.Near()
    #电子图案计算三角形斜边的角度
    for i in range(0, 7):
        pattern.angel_hypotenuse =angle(pattern.vertex[i][0], pattern.vertex[i][1],1)
        pattern.no_shape[str(i)].append(pattern.angel_hypotenuse)
        """
    for i in range(0, 7):
        print(pattern.no_shape[str(i)])
        """
    end = time.clock()

    main()

    for i in range(0, 7):
        if pattern.no_shape[str(i)][4] == False:
            bfs(pattern.graph,i,no_shape,vertex,pattern)
    for i in range(0,7):
        print(no_shape[str(i)])

    image.show()
    name=random.randint(0, 30)
    image.save("pictures/output/"+str(name)+".jpg")
