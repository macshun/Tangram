# 识别目标图案，分割识别出其中用到了哪些简单多边形及其它们相邻的边
# -----------------------------#
# 对每个轮廓遍历轮廓的像素点，若存在相同的则相邻
# -----------------------------#
# 实物图先旋转到目标图案的角度，再寻找相应顶点对应的边
# ---------------------------------#
# 实现邻边，顶点相邻的识别，用图的结构来存储
import collections
import cv2
import numpy as np
import time
import object
# 数据结构存储图的单个点信息，包括
# A.no(图形A相邻接图形B的编号)
# A.own(A邻接部分的边或顶点)
# A.near(B邻接部分的边或顶点)

class Patch():
    pass


# 图的数据结构，存储图形的邻边、顶点(Patch对象）
graph = {
    '0': [],
    '1': [],
    '2': [],
    '3': [],
    '4': [],
    '5': [],
    '6': []
}
# 存储每个图形的形状、颜色、中心点、是否已经拼凑
no_shape = {
    '0': [],
    '1': [],
    '2': [],
    '3': [],
    '4': [],
    '5': [],
    '6': []
}
# 轮廓上的所有点
cnt_point = []
# 图形的顶点
vertex = []
# 存储A与B邻边的所有像素点
# near[0]中存储m个list，每个list中存储n个像素点
# m=邻接的边数，n=构成每条邻边的所有像素点
near = []
# 初始化
for i in range(0, 7):
    cnt_point.append([])
    # vertex对每个图形存储对应顶点编号，如vertex[0][0]=(x,y)
    # vertex[0]第0个图形，vertex[0][0]第0个图形的第0个顶点
    vertex.append([])
    near.append([])


# 主函数
def main():
    start = time.clock()
    src = cv2.imread("image/mould/18.jpg")
    # 当目标图案过大时进行压缩，指定大小而不是以比例压缩（否则会影响后期像素点检测）
    size = src.shape
    # src = cv2.resize(src, (980,875), interpolation=cv2.INTER_LANCZOS4)
    #src = contrast_brightness_image(src, 1.2, 1)
    #cv2.imshow("sa", src)
    #cv2.waitKey(0)
    ld = ShapeAnalysis()
    ld.analysis(src)
    Near()
    print("t1")
    for i in range(0, 7):
        if len(vertex[i]) == 3:
            angel_hypotenuse = object.angle(vertex[i][0], vertex[i][1],0)
            no_shape[str(i)].append(angel_hypotenuse)
    end = time.clock()
    print("final is in ",end-start)
# 并查集
# 存储图形的根结点/集合
father = [0] * 7
# 是否为根结点
isRoot = [0] * 7
def findFather(x):
    a = x
    while x != father[x]:
        x = father[x]
    while a != father[a]:
        z = a
        a = father[a]
        father[z] = x
    return x
def Union(a, b):
    faA = findFather(a)
    faB = findFather(b)
    if faA != faB:
        father[faA] = faB
def init(n):
    for i in range(0, n):
        father[i] = i
        isRoot[i] = 0
def num_Union(graph):
    init(7)
    ans = 0
    for i in range(0, 7):
        for j in range(0, len(graph[str(i)])):
            no = graph[str(i)][j].no
            #print("cc" + str(no))
            Union(i, no)
    for i in range(0, 7):
        isRoot[findFather(i)] = 1
    for i in range(0, 7):
        if isRoot[i] == True:
            ans += 1
    return ans
init(7)
#寻找边与边相邻
def Part(vertex,index,k,near,i,ll):
    # 将斜率与两个图形进行对比，分别找到它们对应的边
    k2=slope(vertex[index][0], vertex[index][1])
    k3=slope(vertex[index][1], vertex[index][2])
    # 图形有两类，三角形、四边形
    if len(vertex[index])==4:
        k4 = slope(vertex[index][2], vertex[index][3])
        k5 = slope(vertex[index][0], vertex[index][3])
    else:
        k4 = slope(vertex[index][0], vertex[index][2])
    #当存在两条斜率相同的边时，以距离检测到的像素点远近来区别
    # 斜率为无穷时，只需要根据x值来判断
    if k == float("inf"):
        r = near[i][ll][0].x
    else:
        #用于求直线方程
        b = -1 * near[i][ll][0].y - k * near[i][ll][0].x
        #print("b:" + str(b))
    if len(vertex[index])==4:
        if k== float("inf"):
            if k == k2 == k4:
                if abs(vertex[index][1].x - r) < abs(vertex[index][3].x - r):
                    temp= (0, 1)
                else:
                    temp= (2, 3)
            elif k == k3 == k5:
                if abs(vertex[index][1].x - r) < abs(vertex[index][3].x - r):
                    temp = (1, 2)
                else:
                    temp = (0, 3)
        else:
            # 顶点编号选了1，3，可从（0，1）（2，3）中各任选一个
            # 已知直线的方程，带入顶点1，3的x求y,当得到的y与实际的y差距最小时，该顶点就是所求的顶点
            r1 = -1 * (k * vertex[index][1].x + b)
            r2 = -1 * (k * vertex[index][3].x + b)
            if k == k2 == k4:
                if abs(vertex[index][1].y - r1) < abs(vertex[index][3].y - r2):
                    temp = (0, 1)
                else:
                    temp = (2, 3)
            elif k == k3 == k5:
                if abs(vertex[index][1].y - r1) < abs(vertex[index][3].y - r2):
                    temp = (1, 2)
                else:
                    temp = (0, 3)
    #针对三角形
    else:
        if k == k3:
            temp = (1, 2)
        elif k == k4:
            temp = (0, 2)
        elif k == k2:
            temp = (0, 1)
    if temp==(0,1):
        angl = object.angle(vertex[index][0], vertex[index][1],1)
        temp=(0,1,angl)
    elif temp==(2,3):
        angl=object.angle(vertex[index][2],vertex[index][3],1)
        temp=(2,3,angl)
    elif temp==(1,2):
        angl=object.angle(vertex[index][1],vertex[index][2],1)
        temp=(1,2,angl)
    elif temp==(0,3):
        angl=object.angle(vertex[index][0],vertex[index][3],1)
        temp=(0,3,angl)
    elif temp==(0,2):
        angl=object.angle(vertex[index][0],vertex[index][2],1)
        temp=(0,2,angl)
    return temp
#点与点相邻
def Part2(graph,vertex,ans):
    for i in range(0, 7):
        #以连通为条件，当图连通时不去判断，不一定使每个图形都包含自己相邻的所有顶点，能保证拼好图形即可
        if ans != 1:
            #print("集合数：" + str(ans) + "," + str(i))
            #图形A仅有1个相邻的或没有时都需要判断是不是有点与点相邻的情况
            if len(graph[str(i)]) !=3 and len(graph[str(i)])!=4:
                for u in range(0, 7):
                    #寻找集合外的图形进行比对
                    if father[u] != father[i]:
                        o = Patch()
                        p = Patch()
                        for j in range(0, len(vertex[i])):
                            for v in range(0, len(vertex[u])):
                                if abs(vertex[i][j].x - vertex[u][v].x) <= 20 and abs(
                                        vertex[i][j].y - vertex[u][v].y) <= 20:
                                    o.no = u
                                    o.own = (j, j)
                                    o.near = (v, v)
                                    p.no = i
                                    p.own = (v, v)
                                    p.near = (j, j)
                                    p.transfer = (v, j, 0, 0)
                                    o.transfer = (j, v, 0, 0)
                                    graph[str(u)].append(p)
                                    graph[str(i)].append(o)
                                    ans=num_Union(graph)
#点与边相邻
def Part3(graph, vertex, cnt_point, i, min_index, p1, p2,flag):
    num = 0
    for vi in range(0, len(vertex[i])):
        kp = 5.6
        for j in range(0, len(cnt_point[min_index])):
            if abs(vertex[i][vi].x - cnt_point[min_index][j].x) <= 25 and abs(
                    vertex[i][vi].y - cnt_point[min_index][j].y) <= 25:
                num += 1
                if num == 1:
                    p1.x = cnt_point[min_index][j].x
                    p1.y = cnt_point[min_index][j].y
                if num == 20:
                    p2.x = cnt_point[min_index][j].x
                    p2.y = cnt_point[min_index][j].y
                    kp = slope(p1, p2)
                    angl=object.angle(p1,p2,1)

        a = 0
        b = 0
        nump = 0
        for m in range(0, len(vertex[min_index])):
            if slope(p1, vertex[min_index][m]) == kp:
                nump += 1
                if nump == 1:
                    a = m
                else:
                    b = m
        if a != 0 or b != 0:
            p = Patch()
            o = Patch()
            o.no = i
            o.own = (a, b,angl)
            o.near = (vi, vi)
            p.no = min_index
            p.own = (vi, vi)
            p.near = (a, b,angl)
            p.transfer = detail_point(i, min_index, p.own, p.near)
            o.transfer = (p.transfer[1], p.transfer[0], -p.transfer[2], -p.transfer[3])
            graph[str(i)].append(p)
            graph[str(min_index)].append(o)
        else:
            flag=False
    return  flag
def Near():
    #1、判断边与边相邻的情况
    # 与A邻接的B的编号
    s = 0
    for i in range(0, 7):
        num = 0
        for u in range(0, 7):
            if i != u:
                list = []
                for j in range(0, len(cnt_point[i])):
                    for v in range(0, len(cnt_point[u])):
                        if cnt_point[i][j].x == cnt_point[u][v].x and cnt_point[i][j].y == cnt_point[u][v].y:
                            list.append(cnt_point[u][v])
                            s = u
                    if len(list) != 0:
                        near[i].append(list)
            # 定义当前形状相邻的边数
            ll = num
            num = len(near[i])
            # 若检测出新的边
            if (num > ll):
                q = Patch()
                q.no = s
                if len(near[i][ll]) != 0:
                    length = len(near[i][ll]) - 1
                    # --------------------------------#
                    # 计算邻边的斜率
                    k1 = slope(near[i][ll][0], near[i][ll][length])
                    q.own=Part(vertex,i,k1,near,i,ll)
                    q.near=Part(vertex,s,k1,near,i,ll)
                    q.transfer = detail_point(i, s, q.own, q.near)
                graph[str(i)].append(q)
    ans=num_Union(graph)
    print("集合数：" + str(ans))
    # 2、判断点与点相邻的情况
    if ans!=1:
        Part2(graph,vertex,ans)
    ans=num_Union(graph)
    print("集合数：" + str(ans))
    # 3、判断点与边相邻的情况
    if ans!=1:
        for i in range(0, 7):
            if len(graph[str(i)]) != 3 and len(graph[str(i)])!=4:
                x1 = no_shape[str(i)][2][0]
                y1 = no_shape[str(i)][2][1]
                min = float("inf")
                min_index = i
                for j in range(0, 7):
                    if father[j] != father[i]:
                        x2 = no_shape[str(j)][2][0]
                        y2 = no_shape[str(j)][2][1]
                        d = pow(x1 - x2, 2) + pow(y1 - y2, 2)
                        if d < min:
                            min = d
                            min_index = j
                # flag标志当前图形是点与边相邻还是边与点相邻
                #True（点与边相邻）
                flag = True

                if min_index!=i:
                    p1=Point()
                    p2=Point()
                    Part3(graph,vertex,cnt_point,i,min_index,p1,p2,flag)
                    if flag==False:
                        Part3(graph,vertex,cnt_point,min_index,i,p1,p2,flag)


    for i in range(0, 7):
        #print("\n")
        print(str(i) + ":" + str(no_shape[str(i)][0]) + "," + str(no_shape[str(i)][1]))
        for j in range(0, len(graph[str(i)])):
            no = graph[str(i)][j].no
            print(str(graph[str(i)][j].no) + ":" + str(no_shape[str(no)][1]), end='')
            print(str(graph[str(i)][j].transfer),end='')
            #print(str(graph[str(i)][j].own) + " " + str(graph[str(i)][j].near) + " " + str(graph[str(i)][j].transfer),
                #  end='')
        print("\n")

#求具体相邻的边的部分
def detail_point(n1,n2,point,edge):
    #n1顶点图形，n2边图形
    #点与边相邻
    #寻找x最小的顶点进行比较
    if vertex[n2][edge[0]].x<=vertex[n2][edge[0]].x:
        min=edge[0]
    else:
        min=edge[1]
    #计算差值与移动
    #是否找到重合顶点
    iscoincide=False
    gap_y=vertex[n1][point[0]].y-vertex[n2][min].y
    gap_x = vertex[n1][point[0]].x - vertex[n2][min].x
    for i in range(0,2):
        for j in range(0,2):
            if abs(vertex[n1][point[i]].x-vertex[n2][edge[j]].x)<=15 and abs(vertex[n1][point[i]].y-vertex[n2][edge[j]].y)<=15:
                transfer=(point[i],edge[j],0,0)
                iscoincide=True
    if iscoincide==False:
        #gap_x>15向右，gap_y>15向下
        """
        if (gap_x>15 or gap_x<-15) and (gap_y>15 or gap_y<-15):
            transfer = (point[0], min, gap_x, gap_y)
        else:
            transfer=(point[0],min,0,0)

            if gap_x>15:
                print(str(n1)+" "+str(point[0])+","+"相对于"+str(n2)+" "+str(min)+"向右移动"+str(gap_x))
            elif gap_x<-15:
                print(str(n1) + " " + str(point[0]) + "," + "相对于" + str(n2) + " " + str(min) + "向左移动" + str(-gap_x))
            if gap_y > 15:
                print(str(n1) + " " + str(point[0]) + "," + "相对于" + str(n2) + " " + str(min) +"向下移动" + str(gap_y))
            elif gap_y<-15:
                print("向上移动" + str(-gap_y))
"""
        if -15<=gap_x<=15 and -15<=gap_y<=15:
            transfer=(point[0],min,0,0)
        elif -15<=gap_x<=15:
            transfer=(point[0],min,0,gap_y)
        elif -15<=gap_y<=15:
            transfer=(point[0],min,gap_x,0)
        else:
            transfer=(point[0],min,gap_x,gap_y)

    return transfer
def slope(p1, p2):
    if 0 <= abs(p1.x - p2.x) <= 15 and p1.y != p2.y:
        slope = float("inf")
    else:
        # 邻边斜率
        slope = ((-1) * (p1.y) + p2.y) / (p1.x - p2.x)

        if (abs(slope) < 0.2):
            slope = 0
        if abs(abs(slope) - 1) < 0.2:
            if slope < 0:
                slope = -1
            else:
                slope = 1

    return slope
# 数组倒序
def Reverse(vertex, k, m):
    # N=len(vertex)
    j = m
    for i in range(k, k + int((m - k + 1) / 2)):
        print("kd")
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
        # print(str(p1.x) + str(p2.x) + str(p3.x))
        return p3
    elif abs(d1 - d3) < abs(d1 - d2) and abs(d1 - d3) < abs(d2 - d3):
        # print(str(p1.x) + str(p2.x) + str(p3.x))
        return p1
    else:
        # print(str(p1.x) + str(p2.x) + str(p3.x))
        return p2



class Point():
    x = 0
    y = 0

def getColorList():
    # 定义字典存放颜色分量上下限
    # 例如：{颜色: [min分量, max分量]}
    # {'red': [array([160,  43,  46]), array([179, 255, 255])]}
    dict = collections.defaultdict(list)

    # 黑色

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
    dict['gray'] = color_list

    # 白色
    # lower_white = np.array([0, 0, 221])
    # upper_white = np.array([180, 30, 255])
    # color_list = []
    # color_list.append(lower_white)
    # color_list.append(upper_white)
    # dict['white'] = color_list

    #粉色
    lower_pink=np.array([160,86,178])
    upper_pink=np.array([180,165,255])
    color_list=[]
    color_list.append(lower_pink)
    color_list.append(upper_pink)
    dict['pink']=color_list

    # 红色

    lower_red = np.array([160, 166, 80])
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

    # 橙色
    lower_orange = np.array([6, 43, 46])
    upper_orange = np.array([24, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # 黄色
    lower_yellow = np.array([25, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # 绿色

    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # 青色

    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([104, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    # 蓝色

    lower_blue = np.array([105, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色

    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([159, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list

    return dict


# 对比度调整
def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道

    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如下
    #cv2.imshow("con-bri-demo", dst)
    return dst


class ShapeAnalysis:
    def __init__(self):
        self.shapes = {'triangle': 0, 'square': 0, 'polygons': 0, 'circles': 0, 'parallelogram': 0}

    def draw_text_info(self, image):
        c1 = self.shapes['triangle']
        c2 = self.shapes['square']
        c3 = self.shapes['polygons']
        c4 = self.shapes['circles']
        c5 = self.shapes['parallelogram']
        cv2.putText(image, "triangle: " + str(c1), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv2.putText(image, "square: " + str(c2), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv2.putText(image, "parallelogram: " + str(c5), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        return image

    def analysis(self, frame):
        #表示第几个图形
        k = 0
        print('go in get_color')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        maxsum = -100
        color = None
        color_dict = getColorList()
        h, w, ch = frame.shape
        result = np.zeros((h, w, ch), dtype=np.uint8)
        for d in color_dict:
            mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])

            mask = cv2.bilateralFilter(mask, 9, 75, 75)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            closed = cv2.erode(closed, None, iterations=5)
            mask = cv2.dilate(closed, None, iterations=5)
            print("start to detect lines...\n")
            out_binary, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # out_binary, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.imshow(d,out_binary)
            #cv2.waitKey(0)
            # 轮廓个数，正确的是一条轮廓线
            if len(contours) == 1:

                # 轮廓上的像素点数
                # print(len(contours[0]))
                # print(contours[0])
                # 轮廓上的每个像素点[]
                # print(contours[0][2][0])

                for cnt in range(len(contours)):
                    # 提取与绘制轮廓
                    cv2.drawContours(result, contours, cnt, (0, 255, 0), 1)
                    # 轮廓逼近
                    epsilon = 0.05 * cv2.arcLength(contours[cnt], True)
                    approx = cv2.approxPolyDP(contours[cnt], epsilon, True)
                    # 图形顶点
                    for i in range(0, len(approx)):
                        p = Point()
                        p.x = approx[i][0][0]
                        p.y = approx[i][0][1]
                        vertex[k].append(p)
                        cv2.circle(result, (approx[i][0][0], approx[i][0][1]), 2, (255, 0, 0), 2)
                    # 分析几何形状
                    corners = len(approx)
                    shape_type = ""
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
                        w = rect[1][0]
                        h = rect[1][1]
                        ar = w / float(h)
                        #print(ar)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(result, [box], 0, (0, 0, 255), 1)
                        # cv2.imshow("ss", image)
                        # cv2.waitKey(0)
                        if ar >= 0.80 and ar <= 1.20:
                            count = self.shapes['square']
                            count = count + 1
                            self.shapes['square'] = count
                            shape_type = "square"
                            # 正方形编号
                            min = Xmin(vertex[k])

                        else:
                            count = self.shapes['parallelogram']
                            count = count + 1
                            self.shapes['parallelogram'] = count
                            shape_type = "parallelogram"
                            # print("ko"+str(vertex[k][0].x)+" "+str(vertex[k][0].y))
                            # print(box[0][0])
                            #对平行四边形进行编号（与box重合的点为0）
                            break_flag = False
                            while break_flag == False:
                                for i in range(0, 4):
                                    while break_flag == False:
                                        for j in range(0, 4):
                                            if abs(vertex[k][i].x - box[j][0]) <= 1 and abs(
                                                    vertex[k][i].y - box[j][1]) <= 1:
                                                min = i
                                                print(min)
                                                break_flag = True
                        if min != 0:
                            Reverse(vertex[k], min, 3)
                            Reverse(vertex[k], 0, min - 1)
                            Reverse(vertex[k], 0, 3)
                            print(str(vertex[k][0].x) + "," + str(vertex[k][0].y))
                        else:
                            print("ok")

                    if corners >= 10:
                        count = self.shapes['circles']
                        count = count + 1
                        self.shapes['circles'] = count
                        shape_type = "circle"
                    if 4 < corners < 10:
                        count = self.shapes['polygons']
                        count = count + 1
                        self.shapes['polygons'] = count
                        shape_type = "polygon"

                    # 求解中心位置
                    print(len(contours))
                    mm = cv2.moments(contours[cnt])
                    cx = int(mm['m10'] / mm['m00'])
                    cy = int(mm['m01'] / mm['m00'])
                    cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

                    # 颜色分析
                    color = frame[cy][cx]
                    color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"
                    cv2.putText(result, str(d) + " " + shape_type, (cx - 20, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    no_shape[str(k)].append(shape_type)
                    no_shape[str(k)].append(d)
                    no_shape[str(k)].append((cx, cy))
                    #no_shape[str(k)].append(False)
                    # print(no_shape[str(k)][2][0])
                    for i in range(0, len(contours[0])):
                        p = Point()
                        p.x = contours[0][i][0][0]
                        p.y = contours[0][i][0][1]
                        p.shape = shape_type
                        p.color = d
                        cnt_point[k].append(p)
                    #print("sssks" + str(len(cnt_point[k])))
                    # 计算面积与周长
                    p = cv2.arcLength(contours[cnt], True)

                    area = cv2.contourArea(contours[cnt])
                    #print("周长: %.3f, 面积: %.3f 颜色，: %s 形状: %s " % (p, area, color_str, shape_type))
                    k += 1
                    #print(k)
        cv2.imshow("Analysis Result", self.draw_text_info(result))
        cv2.imwrite("image/t.jpg", result)
        cv2.waitKey(0)
        #cv2.imwrite("D:/test-result.png", self.draw_text_info(result))
        return self.shapes

if __name__ == '__main__':
    main()

