import numpy as np
import imutils
import cv2
from PIL import Image
import collections
import colorsys
import glob
import os
import time
from tools import *
# from analysis import *
'''
#rgb转hsv

def rgb_hsv(x,y,image):
    h,w=image.size
    color = image.getpixel((x, y))
    new_color = colorsys.rgb_to_hsv(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    #print(new_color)
    hsv = [int(new_color[0] * 360 / 2), int(new_color[1] * 255), int(new_color[2] * 255)]
    return  hsv

#处理高光
def Range(x,y,image):
    new_colors=rgb_hsv(x,y,image)
    #print(new_colors)
    dict=getColorList()
    if dict['white'][0][0] <= new_colors[0] <= dict['white'][1][0] and dict['white'][0][1] <= \
            new_colors[1] <= dict['white'][1][1] and dict['white'][0][2] <= new_colors[2] <= \
            dict['white'][1][2]:
        return True
    elif dict['gray'][0][0] <= new_colors[0] <= dict['gray'][1][0] and dict['gray'][0][1] <= \
            new_colors[1] <= dict['gray'][1][1] and dict['gray'][0][2] <= new_colors[2] <= \
            dict['gray'][1][2]:
        return True
    else:
        return False

#扫描线种子填充算法
def scanFill(x,y,newcolor,image):
    #种子点入栈
    stack=[]
    value=0
    #print((x,y))
    if Range(x,y,image)==True:
        #print((x,y))
        #print(image.getpixel((x,y)))
        #存放当前种子
        seed=[]
        seed.append(x)
        seed.append(y)
        stack.append((x,y))
        #种子颜色，即待填充颜色
        #color=[]

        #print(color[0])
        while len(stack)!=0:
            # 出栈一个种子点
            seed=stack.pop()
            #print("种子",seed)
            x=seed[0]
            y=seed[1]
            #color.append(image.getpixel((x-200, y)))
            #向右填充
            while Range(x,y,image)==True:
                image.putpixel((x,y),newcolor)
                x+=1
                #print("11 "+str(x)+","+str(y))
            xr=x-1
            #向左填充
            if seed[0]>0:
                x=seed[0]-1
            while Range(x,y,image)==True :
                image.putpixel((x, y), newcolor)
                x-=1
                #print("22 "+str(x)+","+str(y))
            xl=x+1
            #至此找到xl和xr0
            #处理上一条扫描线y-1
            x=xl
            y=y-1
            while x<xr:
                spanNeedFill=False
                #找到最右边界点xr作为新种子
                while Range(x,y,image)==True:
                    spanNeedFill=True
                    x+=1
                #新种子入栈
                if spanNeedFill:
                    #change(seed,x,y)
                    seed=[x-1,y]
                    stack.append(seed)
                    spanNeedFill=False
                    #print("33",seed)
                #寻找需要填充的第一个像素点||都不需要填充
                while Range(x,y,image)==False and x<xr:
                    x+=1
            #处理下一条扫描线y+2
            x=xl
            y=y+2
            while x<xr:
                spanNeedFill=False
                while Range(x,y,image)==True:
                    spanNeedFill=True
                    x+=1
                while spanNeedFill:
                    #change(seed,x,y)
                    seed = [x - 1, y]
                    stack.append(seed)
                    spanNeedFill=False
                    #print("44",seed)
                while Range(x,y,image)==False and x<xr:
                    x+=1
        value=1
    return value
#筛查颜色
def Color(x,y,image):
    new_colors=rgb_hsv(x,y,image)
    print(new_colors)
    dict=getColorList()
    for i in dict:
        if dict[i][0][0] <= new_colors[0] <= dict[i][1][0] and dict[i][0][1] <= \
                new_colors[1] <= dict[i][1][1] and dict[i][0][2] <= new_colors[2] <= \
                dict[i][1][2]:
            #print("hkig"+str(i))
            return i
        if len(dict[i])==4:
            if dict[i][2][0] <= new_colors[0] <= dict[i][3][0] and dict[i][2][1] <= \
                    new_colors[1] <= dict[i][3][1] and dict[i][2][2] <= new_colors[2] <= \
                    dict[i][3][2]:
                # print("hkig"+str(i))
                return i

def SecondFill(x,y,image):
    if Color(x,y,image)=='white':
        print("white")
#hsv颜色字典
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
    #lower_pink=np.array([166,86,178])
    #upper_pink=np.array([180,165,255])
    lower_pink=np.array([170,70,230])
    upper_pink=np.array([180,130,255])
    lower_pink1=np.array([0,76,170])
    upper_pink1=np.array([7,140,255])
    color_list=[]
    color_list.append(lower_pink)
    color_list.append(upper_pink)
    color_list.append(lower_pink1)
    color_list.append(upper_pink1)
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
    upper_red = np.array([6, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list

    #橙色
    lower_orange = np.array([7, 102, 153])
    upper_orange = np.array([15, 230, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    #黄色
    lower_yellow = np.array([25, 43, 180])
    upper_yellow = np.array([33, 255, 255])
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

    lower_purple = np.array([125, 76, 115])
    upper_purple = np.array([180, 153, 204])
    #lower_purple1=np.array([170,])
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
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如下
    #cv2.imshow("con-bri-demo", dst)
    #cv2.waitKey(0)
    return  dst


def ostu(img):
    area=0
    image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转灰度
    blur = cv2.GaussianBlur(image,(5,5),0) # 阈值一定要设为 0 ！高斯模糊
    ret3,th3 = cv2.threshold(blur,60,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # 二值化 0 = black ; 1 = white
    #cv2.imshow('image', th3)
    #a = cv2.waitKey(0)
    # print a
    #print(th3[355,16])
    #print("hhh")
    height, width = th3.shape
    for i in range(height):
        for j in range(width):
            if th3[i, j]==255:
                area+=1
                if area==5:
                    return i,j
'''

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
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #cv2.imshow("blurred",blurred)
        #cv2.waitKey(0)
        thresh = cv2.threshold(blurred, 60,255, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.erode(closed, None, iterations=2)
        thresh = cv2.dilate(closed, None, iterations=1)
"""
        #thresh=cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1)
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
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
        # cv2.imshow("blurred",blurred)
        # cv2.waitKey(0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=2)
        thresh = cv2.dilate(closed, None, iterations=2)
        #cv2.imshow("OpenCV",thresh)
        #cv2.waitKey(0)

        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        print(len(cnts))
        # loop over the contours

        for c in cnts:
            #print(c)
            # 轮廓逼近
            epsilon = 0.05 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            # 分析几何形状
            corners = len(approx)
            print("KKK" + str(corners))
            shape_type = ""
            if corners == 3:
                count = self.shapes['triangle']
                count = count + 1
                self.shapes['triangle'] = count
                shape_type = "triangle"
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
                else:
                    count=self.shapes['parallelogram']
                    count=count+1
                    self.shapes['parallelogram']=count
                    shape_type="parallelogram"
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
            #print(M)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            color=Color(cX,cY,im)
            #print(color)
            # draw the contour and center of the shape on the image
            cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
            cv2.putText(image, str(color)+" "+shape_type, (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #cv2.imshow("s",image)
            #cv2.waitKey(0)
        cv2.imshow("Analysis Result", self.draw_text_info(image))
        cv2.waitKey(0)

def main():
    start=time.clock()
    #image = cv2.imread("image/shoot/no_highlight/(75).jpg")

    for i in range(1,50):
        # strs="image/pictures/"+str(i)+".jpg"
        strs="pictures/"+str(i)+".jpg"

        print(strs)
        # image=cv2.imread("image/pictures/"+str(i)+".jpg")
        image=cv2.imread(strs)

        size=image.shape
        img = cv2.resize(image, (int(size[1]),int(size[0])), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]  # 获取图像的高和宽
        #cv2.imshow("Origin", img)  # 显示原始图像
        #cv2.waitKey(0)
        #存在高光时，适量降低图片亮度
        #img = np.uint8(np.clip((1.5 * img),0 , 255))
        img = contrast_brightness_image(img, 1.5,-20)
        cv2.imwrite("image/yu.jpg",img)
        #cv2.imshow("contrast",img)
        #cv2.waitKey(0)
        blured = cv2.GaussianBlur(img, (5, 5), 0)
        #blured = cv2.blur(img, (5, 5))  # 进行滤波去掉噪声
        blured_copy=blured.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)  # 掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
        # 进行泛洪填1
        u, v = ostu(blured)
        #cv2.floodFill(blured, mask, (v, u), (0, 0, 0), (100, 100, 100), (20, 20, 20), cv2.FLOODFILL_FIXED_RANGE)
        cv2.floodFill(blured, mask, (w - 1, h - 1), (0, 0, 0), (50, 35, 50), (185, 190, 190), cv2.FLOODFILL_FIXED_RANGE)
        #cv2.floodFill(blured, mask, (20, 20), (0, 0, 0), (50,50,50), (0, 0, 0), cv2.FLOODFILL_FIXED_RANGE)
        #cv2.floodFill(blured, mask, (0, 0), (0,0,0), (0,0,0), (255,255,255), cv2.FLOODFILL_FIXED_RANGE)
        cv2.imshow("floodfill1", blured)
        cv2.waitKey(0)
        ld = Analysis()
        ld.analy(blured)
        if ld.shapes['triangle']==5 and ld.shapes['parallelogram']==1 and ld.shapes['square']==1:
            print('yes')
        else:
            ld.shapes['triangle'] =0
            ld.shapes['parallelogram'] =0
            ld.shapes['square'] =0
            ld.analy(blured_copy)
        end=time.clock()
        print("final is in ",end-start)

    """
    image=cv2.imread("image/pictures/"+str(37)+".jpg")
    size=image.shape
    img = cv2.resize(image, (int(size[1]),int(size[0])), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]  # 获取图像的高和宽
    cv2.imshow("Origin", img)  # 显示原始图像
    cv2.waitKey(0)
    #存在高光时，适量降低图片亮度
    #img = np.uint8(np.clip((1.5 * img),0 , 255))
    img = contrast_brightness_image(img, 1.5,-20)
    cv2.imwrite("image/yu.jpg",img)
    #cv2.imshow("contrast",img)
    #cv2.waitKey(0)
    blured = cv2.GaussianBlur(img, (5, 5), 0)
    #blured = cv2.blur(img, (5, 5))  # 进行滤波去掉噪声
    blured_copy=blured.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)  # 掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
    # 进行泛洪填1
    u, v = ostu(blured)
    #cv2.floodFill(blured, mask, (v, u), (0, 0, 0), (100, 100, 100), (20, 20, 20), cv2.FLOODFILL_FIXED_RANGE)
    cv2.floodFill(blured, mask, (w - 1, h - 1), (0, 0, 0), (50, 35, 50), (185, 190, 190), cv2.FLOODFILL_FIXED_RANGE)
    #cv2.floodFill(blured, mask, (20, 20), (0, 0, 0), (50,50,50), (0, 0, 0), cv2.FLOODFILL_FIXED_RANGE)
    #cv2.floodFill(blured, mask, (0, 0), (0,0,0), (0,0,0), (255,255,255), cv2.FLOODFILL_FIXED_RANGE)
    cv2.imshow("floodfill1", blured)
    cv2.waitKey(0)
    ld = Analysis()
    ld.analy(blured)
    if ld.shapes['triangle']==5 and ld.shapes['parallelogram']==1 and ld.shapes['square']==1:
        print('yes')
    else:
        ld.shapes['triangle'] =0
        ld.shapes['parallelogram'] =0
        ld.shapes['square'] =0
        ld.analy(blured_copy)
    end=time.clock()
    print("final is in ",end-start)
    """
if __name__ == '__main__':
    main()