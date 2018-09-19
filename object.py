import numpy as np
import imutils
import cv2
from PIL import Image
import collections
import colorsys
import time
import math
import pattern
from PIL import ImageDraw
import random
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


class Point():
    x = 0
    y = 0
# 初始化
for i in range(0, 7):
    cnt_point.append([])
    # vertex对每个图形存储对应顶点编号，如vertex[0][0]=(x,y)
    # vertex[0]第0个图形，vertex[0][0]第0个图形的第0个顶点
    vertex.append([])
# 生成深蓝色绘图画布
array = np.ndarray((1190, 1190, 3), np.uint8)
array[:, :, 0] = 0
array[:, :, 1] = 0
array[:, :, 2] = 100
image = Image.fromarray(array)
# 创建绘制对象
draw = ImageDraw.Draw(image)


#rgb转hsv
def rgb_hsv(x,y,image):
    color = image.getpixel((x, y))
    new_color = colorsys.rgb_to_hsv(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    hsv = [int(new_color[0] * 360 / 2), int(new_color[1] * 255), int(new_color[2] * 255)]
    return  hsv


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


#hsv颜色字典： Hue-色调、Saturation-饱和度、Value-值s
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


#亮度、对比度调整
def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    #dst = src1*alpha + src2*beta + gamma
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如下
    #cv2.imshow("con-bri-demo", dst)
    #cv2.waitKey(0)
    return  dst


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

#某一点逆时针旋转theta（弧度）
def rotatePoint(p1,p,theta):
    p_r=Point()
    p_r.x=p.x+(p1.x-p.x)*math.cos(theta)-(p1.y-p.y)*math.sin(theta)
    p_r.y=p.y+(p1.x-p.x)*math.sin(theta)+(p1.y-p.y)*math.cos(theta)
    return p_r
# 数组倒序
def Reverse(vertex, k, m):
    # N=len(vertex)
    j = m
    for i in range(k, k + int((m - k + 1) / 2)):
        u = vertex[j]
        vertex[j] = vertex[i]
        vertex[i] = u
        j -= 1
# 寻找正方形x最小的点的下标
def Xmin(vertex):
    l = len(vertex)
    min = 0
    for i in range(0, l):
        if vertex[i].x < vertex[min].x:
            min = i
    return min
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
#由斜率求角度
def angle(p1,p2,flag):
    temp1 = slope(p1, p2,flag)
    temp1 = math.atan(temp1)
    angle = math.degrees(temp1)
    angle=round(angle)
    if angle<0:
        angle=180+angle
    return angle


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


def draw_test(p1,p2,p3,draw,color):
    # 绘制多边形
    draw.polygon((p1.x,p1.y,p2.x,p2.y,p3.x,p3.y), color)
    return


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
                # cv2.inRange(hsv, lower_red, upper_red):
                #   hsv指的是原图
                #   lower_red指的是图像中低于这个lower_red的值，图像值变为0
                #   upper_red指的是图像中高于这个upper_red的值，图像值变为0
                # 而在lower_red～upper_red之间的值变成255
                mask = cv2.inRange(hsv, dict[d][0], dict[d][1])
                #cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) ：
                # dst = src1 * alpha + src2 * beta + gamma;
                #   src1 – first input array.
                #   alpha – weight of the first array elements.
                #   src2 – second input array of the same size and channel number as src1.
                #   beta – weight of the second array elements.
                #   dst – output array that has the same size and number of channels as the input arrays.
                #   gamma – scalar added to each sum.
                #   dtype – optional depth of the output array; when both input arrays have the same depth,
                #         dtype can be set to -1, which will be equivalent to src1.depth().
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
                box = cv2.boxPoints(rect) #获取矩形四个顶点坐标
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
            color=Color(cX,cY,im)
            # draw the contour and center of the shape on the image
            cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
            cv2.putText(image, str(color)+" "+shape_type, (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            no_shape[str(k)].append(shape_type)
            no_shape[str(k)].append(color)
            no_shape[str(k)].append((cX, cY))
            for i in range(0,len(vertex[k])):
                cv2.putText(image,str(i),(vertex[k][i].x,vertex[k][i].y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            k+=1
        cv2.imshow("Analysis Result", self.draw_text_info(image))
        cv2.imwrite("image/pattern.jpg",image)
        cv2.waitKey(0)


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
    image = cv2.imread("pictures_0919/2.jpg")

    size=image.shape
    img = cv2.resize(image, (int(size[1]),int(size[0])), interpolation=cv2.INTER_AREA)
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
    # cv2.waitKey(0)
    blured_copy=blured.copy()
    # mask：掩码图像，大小比原图多两个像素点。设输入图像大小为width * height, 则掩码的大小必须为 (width+2) * (height+2) ,
    #       mask可为输出，也可作为输入 ，由flags决定。
    mask = np.zeros((h + 2, w + 2), np.uint8)  # 掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
    # 进行泛洪填1
    cv2.floodFill(blured, mask, (w - 1, h - 1), (0, 0, 0), (50,35,50), (185, 190, 190), cv2.FLOODFILL_FIXED_RANGE)
    # test for flood fill
    cv2.imshow('flood fill', blured)
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
