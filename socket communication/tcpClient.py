# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-03-28-7:55 下午
# import socket
#
# def tcpClient():
#     host = "127.0.0.1"
#     port = 5000
#
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     sock.connect((host, port))
#
#     message = input("->")
#     while message != 'q':
#         sock.send(message.encode())
#         data = sock.recv(1024)
#         print("received from server "+str(data))
#         message = input("->")
#
#     sock.close()
#
# if __name__=="main":
#     tcpClient()

import socket
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.connect(('127.0.0.1', 8888))
while True:
    data = input('请输入要发送的数据：')
    if data == 'quit':
        break
    serversocket.send(data.encode())
    print(serversocket.recv(1024).decode('utf-8'))
serversocket.send(b'quit')
serversocket.close()
