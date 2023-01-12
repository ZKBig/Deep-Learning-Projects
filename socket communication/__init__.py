# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-03-28-7:28 下午
import numpy as np
import socket
import time
import os
from datetime import datetime
from threading import Thread
from operator import methodcaller

import socket

def tcpServer():
    host = "127.0.0.1"
    port = 5000

    sock = socket.socket()
    sock.bind((host, port))
    sock.listen(1)

    server, address = sock.accept()
    print("Connect from ", str(address))
    while True:
        data = server.recv(1024)
        if not data:
            print("receive failed.")
            break
        print("from connected user ", str(data))
        data = "OK"
        server.send(data.encode())
    server.close()

def tcpClient():
    host = "127.0.0.1"
    port = 5000

    sock =socket.socket()
    sock.connect((host, port))

    message = input("->")
    while message != 'q':
        sock.send(message.encode())
        data = sock.recv(1024)
        print("received from server "+str(data))
        message = input("->")

    sock.close()




