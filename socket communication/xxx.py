# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-03-28-7:35 下午
import numpy as np
import socket
from datetime import datetime
from threading import Thread
import time
import os
from operator import methodcaller


class CPhone:
    def __init__(self):
        self.g_conn_pool = []
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.thread = Thread(target=self._accept_client) #主线程
        self.thread.setDaemon(True)

    def run(self,ip="localhost",port=8888,max_n=0):
        self.server.bind((ip, port))
        self.server.listen(max_n)
        self.thread.start()

        log = """--------------------------\n--------------------------
%s\t\n(ip=%s,port=%d)\n\tstart listening...
"""% (str(datetime.now()), ip, port)
        if os.path.exists('./log.txt'):
            with open('./log.txt','a',encoding='utf-8') as fo:
                fo.write(log)
        else:
            with open('./log.txt','w',encoding='utf-8') as fo:
                fo.write(log)
        print(log)
        while True:
            cmd = input("""--------------------------
    输入1:查看当前在线人数
    输入0:关闭服务器
--------------------------
""")
            if cmd == '1':
                print("--------------------------")
                print("当前在线人数：\n", len(self.g_conn_pool))
            elif cmd == '0':
                exit()

    def _data_handle(self,client,address): #此函数主要用于解析数据，调用函数处理，并将结构序列化为字符串
        func_name,mat = self._recvall(client)
        #cv2.imwrite('xx.jpg', mat)

        return_data = methodcaller(func_name,mat)(self)

        shape_str = [str(i) for i in return_data.shape]
        shape_str = ','.join(shape_str)
        data_str = str(return_data.reshape(-1).tolist()).replace(' ', '').replace('[', ',').replace(']', ',')
        shape_str = ',%s,%d,' % (shape_str, len(data_str))
        self._sendall(client, shape_str, data_str)
        time.sleep(2)
        client.close()
        self.g_conn_pool.remove(client)
        log = ('%s\t用户(ip:%s port:%d)\t已下线\n'% (str(datetime.now()),address[0], address[1]))
        with open('./log.txt', 'a',encoding='utf-8') as fo:
            fo.write(log)
        print(log)


    def recvall(self,sock):
        header = sock.recv(20)
        header = str(header).split('\\',1)[0].replace("b'",'')
        func_name,channels,height,width = header.split(',')
        channels,height,width = int(channels),int(height),int(width)
        count = channels*height*width
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        data = np.frombuffer(buf, dtype='uint8')
        mat = data.reshape(height, width, channels)
        return func_name,mat

    def _sendall(self,sock,header_str,data_str):
        sock.send(bytes(header_str, encoding="ascii"))
        data_bt = bytes(data_str, encoding="ascii")
        total = len(data_str)
        sended = 0
        while sended<total:
            send = min(total-sended,128*128*3)
            sock.send(data_bt[sended:sended+send])
            sended += send

    def _accept_client(self): #此函数用于接收客户端的链接请求，并为其开启一个线程
        while True:
            client, address = self.server.accept()  # 阻塞，等待客户端连接
            # 加入连接池
            self.g_conn_pool.append(client)
            log = ('%s\t用户(ip:%s port:%d)\t已上线\n' % (str(datetime.now()), address[0], address[1]))
            with open('./log.txt','a',encoding='utf-8') as fo:
                fo.write(log)
            print(log)
            # 给每个客户端创建一个独立的线程进行管理
            thread = Thread(target=self._data_handle, args=(client,address))
            # 设置成守护线程
            thread.setDaemon(True)
            thread.start()



