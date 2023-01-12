# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-03-28-7:55 下午
import socket
import time

servesocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
servesocket.bind(('127.0.0.1', 8001))
servesocket.listen(1)
print('Server is running.')
def TCP(sock, addr):
    print('Accept new connection from %s:%s.' % addr)
    while True:
        data = sock.recv(1024)
        time.sleep(1)
        if not data:
            print("Receive data error.")
            break
        print('Receive the data :', data)
        # update the data
        vehicle, new_velocity, new_acceleration = data.decode('utf-8').split(",")
        print(vehicle, new_velocity, new_acceleration)
        new_velocity = float(new_velocity)
        new_acceleration = float(new_acceleration)
        new_velocity += 3.0
        print(new_velocity)
        new_acceleration += 2.0
        print(new_acceleration)
        send_data = str(vehicle)+","+str(new_velocity)+","+str(new_acceleration)
        # send the data to VISSIM
        sock.send(send_data.encode())
    sock.close()
    print('Connection from %s:%s closed.' % addr)
while True:
    sock, addr = servesocket.accept()
    TCP(sock, addr)

