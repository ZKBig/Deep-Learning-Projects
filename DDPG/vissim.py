import socket
# import win32com.client as com 
import os
import random
import socket
import numpy as np

DRIVER_DATA_VEH_DESIRED_VELOCITY=100
DRIVER_DATA_VEH_MAX_ACCELERATION=50

class vissim():
	
	def __init__(self, ip, port):

		# 建立socket.
		assert isinstance(ip, str)
		assert isinstance(port, int)
		self.servesocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.servesocket.bind(('127.0.0.1', 9000))
		self.servesocket.listen(1)
		self.sock, self.addr = self.servesocket.accrpt()
		print('Server is running.')

		self.v = None  # 速度
		self.d = None  # 距离
		self.a = None  # 加速度
		self.state = None;

		
		# 连接Vissim COM Server.
		# self.inpx_file = ""
		# self.layx_file = ""
		# self.Vissim = com.dispatch("Vissim.Vissim.800")
		# # Load the net file.
		# Vissim.LoadNet(inpx_file)
		# # Load the layout file.
		# Vissim.LoadLayout(layx_file)


	# def warmup():
	# 	"""
	# 	form a steady traffic flow and wait for the 20th vehicle.
	# 	"""
		

	def reset(self):
		"""
		renew the simulation
		"""
		print('Accept new connection from %s:%s.' % self.addr)
		# seed = random.randint(1,10000)
		# 1. receive the data length first
		data_length = self.sock.recv(2)
		if not data_length:
			print("Receive data error.")
			return
		print('Receive the data :', data_length)
		data_length = int(data_length.decode('utf-8'))
		print(data_length)
		# 2. receive the data according to the data length
		data = self.sock.recv(data_length)
		if not data:
			print("Receive data error.")
			return
		print('Receive the data :', data)
		# 3. update the data
		lead_vehicle_length, current_speed, current_acceleration = data.decode('utf-8').split(",")
		lead_vehicle_length = float(lead_vehicle_length)
		current_speed = float(current_speed)
		current_acceleration = float(current_acceleration)
		self.state.append(lead_vehicle_length, current_speed, current_acceleration)

		return np.array(self.state)

	def rewardFunction(self, state):
		"""
		Reward function and shaping.
		# """
		inf = 99999999  #
		reward = 0  # 奖励
		safe_d = 2  # 安全距离
		# length = 5  # 车长

		length, current_speed, current_acceleration = state
		self.v = current_speed
		self.a = current_acceleration
		self.d = None
		vm = DRIVER_DATA_VEH_DESIRED_VELOCITY  # 最大速度
		am = DRIVER_DATA_VEH_MAX_ACCELERATION  # 最大加速度

		reward_d_n = -20  # 距离检测消极奖励
		reward_d_p = 5  # 距离检测积极奖励
		reward_v_n = -50  # 速度检测消极奖励
		reward_v_p_k = 0.1  # 速度检测积极奖励系数
		reward_a_n1 = -10  # 加速度检测消极奖励 1
		reward_a_n2 = -5  # 加速度检测消极奖励 2
		reward_a_p = 5  # 加速度检测积极奖励

		# 距离检测
		if self.d <= 0:
			reward += -inf
		else:
			if self.d > 0 and self.d < safe_d:
				reward += reward_d_n
			else:
				reward += reward_d_p

			# 速度检测
		if self.v < 0:
			reward += -inf
		else:
			if self.v == 0:
				reward += reward_v_n
			else:
				if self.v > 0 and self.v <= vm:
					reward += reward_v_p_k * self.v
				else:
					reward += -inf

		# 加速度检测
		if abs(self.a) > abs(am):
			reward += -inf
		else:
			if self.d < safe_d and self.a > 0:
				reward += reward_a_n1
			else:
				if self.d >= safe_d and self.v < vm and self.a < 0:
					reward += reward_a_n2
				else:
					reward += reward_a_p

		if(self.v==0 or self.d<=0):
			done = True

		return reward, done, {}

	def step(self, action):
		"""
		Run every episode (trajectory).
		"""
		lead_vehicle_length, current_speed, current_acceleration = action
		send_data = str(lead_vehicle_length) + "," + str(current_speed) + "," + str(current_acceleration)
		# send the data to VISSIM
		print(send_data)
		self.sock.send(send_data.encode())
		data_length = self.sock.recv(2)
		if not data_length:
			print("Receive data error.")
			return
		print('Receive the data :', data_length)
		data_length = int(data_length.decode('utf-8'))
		print(data_length)
		# receive the data according to the data length
		data = self.sock.recv(data_length)
		if not data:
			print("Receive data error.")
			return
		print('Receive the data :', data)
		# update the data
		lead_vehicle_length, current_speed, current_acceleration = data.decode('utf-8').split(",")
		lead_vehicle_length = float(lead_vehicle_length)
		current_speed = float(current_speed)
		current_acceleration = float(current_acceleration)
		self.state.append(lead_vehicle_length, current_speed, current_acceleration)
		reward, done, info = self.rewardFunction(self.state)

		return np.array(self.state), reward, done, info

