# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-02-26-9:41 下午

import matplotlib.pyplot as plt
import math
i=0
x=[]
y1=[]
y2=[] # 要绘制的是（x,y1）和（x,y2）
# subplot(在窗口中分的行、列、画图序列数)
while (i<10000):
    plt.clf()  # 清除之前画的图
    # subplot(在窗口中分的行、列、画图序列数)
    plt.subplot(211) #第1个图画在一个两行一列分割图的第1幅位置
    x.append(i)
    y1.append(i**2)
    plt.plot(x,y1)
    plt.subplot(212) #第2个图画在一个两行一列分割图的第2幅位置
    y2.append(math.sqrt(i))
    plt.plot(x,y2)
    plt.pause(0.001)  # 暂停0.1秒
    plt.ioff()  # 关闭画图的窗口
    i=i+1
