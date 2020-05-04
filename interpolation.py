# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:21:37 2020

@author: 270
"""


#获取x和y的值
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#获取自变量x及对应的函数值y
n=10
#n = int(input("请输入n值："))
h = 2/n
m = np.arange(0,n+1)
x = -1+m*h
#print(x)

def f(x): #定义函数
    y = 1/(1+15*x**2)
    return y

y = f(x)
#print(y)


#lagrange插值
def Lagrange_interpolation(x,y,x_lag): #x，y为已知数据点，x_lag为待插值横坐标
    y_lag = 0
    for i in range(len(y)):
        t = y[i]
        for j in range(len(y)):
            if i!=j:
                t *= (x_lag-x[j])/(x[i]-x[j]) #先求出每一项，再求和
        y_lag += t
    return(y_lag)

lag = Lagrange_interpolation(x,y,x)
#print(lag)


#分段线性插值
def piecewise_lin_interp(x,y,x_lin):#x，y为已知数据点，x_lin为待插值横坐标
    y_lin = []
    index = -1
    for i in range(len(x_lin)):
        for j in range(len(x)-1):
            #找出P所在的自变量区间,而后进行插值
            if x_lin[i]>=x[j] and x_lin[i] <= x[j+1]:
                y_lin.append((y[j]*((x_lin[i]-x[j+1])/(x[j]-x[j+1]))+y[j+1]*((x_lin[i]-x[j])/(x[j+1]-x[j]))))
    return(y_lin)
       
#print(piecewise_lin_interp(x,y,x))


#三次样条插值
def tdma(a,b,c,d):#追赶法,a,b,c分别对应三对角矩阵系数矩阵系数，a在最下方，b在中间，c在上面，f为齐次项系数
    n = len(d) - 1
    r = np.zeros(len(d))
    y = np.zeros(len(d))
    x = np.zeros(len(d))
    r[0] = c[0]/b[0]
    y[0] = d[0]/b[0]
    for k in range(1,n+1):
        r[k] = c[k]/(b[k]-r[k-1]*a[k])
        y[k] = (d[k] - a[k]*y[k-1])/(b[k]-r[k-1]*a[k])
    #print(r,y)
    x[n] = y[n]
    for i in range(1,n+1):
        x[n-i] = y[n-i] - r[n-i]*x[n-i+1]
    return x

def solution_of_equation(x,y,boundary):#构建三对角矩阵，并进行求解
    n = len(x) - 1 #下标上限
    h = np.zeros(len(x)-1)#h为两个自变量的插值，h[k] = x[k+1] - x[k]
    d = np.zeros(len(x))#齐次项系数
    mu = np.zeros(len(x))#三对角矩阵最下方的对角线系数
    for k in range(n):
        h[k] = x[k+1] - x[k]
    for k in range(1,len(h)):
        mu[k] = h[k-1]/(h[k-1] + h[k])
    mu[n] = 1
    #print(h)
    #print("mu:",mu)
    lamda = 1-mu#三对角矩阵最上方系数
    #print("lamda:",lamda)
    b = np.ones(len(x))*2#三对角矩阵正对角线系数
    #print("b",b)
    for i in range(1,n):
        d[i] = 6*((y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]))/(x[i+1]-x[i-1])
    d[0] = 6.0*((y[1]-y[0])/(x[1]-x[0])-boundary[0])/(h[0])
    d[n] = (6.0/h[n-1])*(boundary[1]-(y[n]-y[n-1])/(x[n]-x[n-1]))
    #print("d",d)
    M = tdma(mu,b,lamda,d)
    return M,h

def sline3_interp(x,y,x_s3):#x，y为已知函数点及其函数值；X为插值点数值
    M,h = solution_of_equation(x,y,boundary)
    #print("M",M)
    #print("h",h)
    y_s3 = np.zeros(len(x_s3))
    for i in range(len(x_s3)):
        for j in range(len(x)-1):
            if x_s3[i]<=x[j+1] and x_s3[i]>=x[j]:
                y_s3[i]=((M[j]*np.power((x[j+1]-x_s3[i]),3))+(M[j+1]*np.power((x_s3[i]-x[j]),3)))/(6*h[j])+(y[j]-(M[j]*np.power(h[j],2)/6))*((x[j+1]-x_s3[i])/h[j])+(y[j+1]-(M[j+1]*np.power(h[j],2)/6))*((x_s3[i]-x[j])/h[j])
    return y_s3


def pic_show(name,x,y):#图片名，插入值x和对应的插值函数值
    fig = plt.figure()
    plt.plot(x, y,"r*-", label='interpolation')
    plt.plot(x, f(x), label='original')
    plt.legend()
    plt.title(name, FontProperties='SimHei')
    plt.savefig(name+'.png')
    plt.close(fig)
    pass
  
    
x_interp= np.linspace(-1,1,100)
y_l = Lagrange_interpolation(x,y,x_interp)
pic_show('lagrange插值与原函数对比图', x_interp, y_l)

y_lin =  piecewise_lin_interp(x,y,x_interp)
pic_show('分段线性插值与原函数对比图', x_interp, y_lin)

y_sline3 = sline3_interp(x,y,x_interp)
pic_show('三次样条插值与原函数对比图', x_interp, y_sline3)

data = pd.DataFrame({"x":x_interp, "y":f(x_interp), "lagrange":y_l, "分段线性":y_lin, "三次样条":y_sline3})
print(data)


plt.plot(x_interp, np.abs(y_l-f(x_interp)), "r+--", label="lagrange(x)")  
plt.plot(x_interp, np.abs(y_lin-f(x_interp)), "g*-",label = "piecewise_lin")     
plt.plot(x_interp, np.abs(y_sline3-f(x_interp)), "k-.",label="sline3_interp")
plt.show
