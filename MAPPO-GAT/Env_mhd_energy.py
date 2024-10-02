import math
import gym
import numpy as np
import random
import torch
random.seed(1103)
np.random.seed(1103)
torch.manual_seed(1103)


class Vehicle:
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity

class AUTOVANET_CV1(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, vari):
        # road model of Manhattan city
        self.down_lanes = [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 450 - 3.5 - 3.5 / 2, 450 - 3.5 / 2]
        self.up_lanes = [50 - 3.5 / 2, 50 + 3.5 / 2, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2]
        self.left_lanes = [50 - 3.5 / 2, 50 + 3.5 / 2, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2]
        self.right_lanes = [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 450 - 3.5 - 3.5 / 2, 450 - 3.5 / 2]
        self.width = 500  # m
        self.height = 500  # m

        self.car_num = vari  # the number of VUE
        self.Distance_limit_rsu = 150  # coverage radius of RSU
        self.vel_m = 25  # average speed of VUE

        self.input_sizes = [6, 8, 10]  # task size
        self.compute_densitys = [400, 500, 600, 700, 800]  # Computational resources required per bit of task
        self.data_min = 6 * 1048576  # Maximum number of bits for a task
        self.data_max = 10 * 1048576  # Minimum number of bits for a task

        self.RSU_d = 6
        self.RSU_num = self.RSU_d ** 2  # the number of RSU
        self.Vehicles = []
        # Define RSU locations
        self.RSU_location = np.zeros([self.RSU_num, 2])
        self.RSU_f_vec = np.zeros([self.RSU_num, ])
        self.RSU_dis = [i * round(500 / (self.RSU_d - 1), 2) for i in range(self.RSU_d)]
        RSU_num = 0
        for i in range(len(self.RSU_dis)):
            for j in range(len(self.RSU_dis)):
                self.RSU_location[RSU_num, 0] = self.RSU_dis[i]
                self.RSU_location[RSU_num, 1] = self.RSU_dis[j]
                RSU_num += 1
        # Computing Resources of RSU
        self.std_f = 10
        for i in range(self.RSU_num):
            self.RSU_f_vec[i] = random.randint(2, 4 + self.std_f)

        self.Pre_RSU_number = 4  # the number of RSUs observable by the VUE
        self.P_j_car = [20 + i for i in range(15)]  # transmit power of VUE
        self.P_j_level = len(self.P_j_car)
        self.P_j_rsu = 40  # transmit power of RSU
        self.sigma2 = 10 ** ((-174 + 5 + 10 * math.log10(180 * 1000)) / 10)  # Noise Power
        self.energy_para = 1e-29
        self.bandwidth = 2 * 1e6  # 2 * 1e6 hz

        # RSU antenna height
        self.h_bs = 1.5
        self.h_ms = 1.5

        # Define MDP
        self.action_dim = self.Pre_RSU_number * self.P_j_level
        self.observation_dim = 3 * self.Pre_RSU_number + 2 + 2
        self.state_dim = self.observation_dim
        self.state = np.zeros([self.car_num, self.observation_dim])
        self.old_action = np.zeros([self.car_num, self.action_dim])

        self.slot = 0.1  # the length of slot
        self.energy_thre = 20  # the threshold of energy consumption
        self.delay_thre = 3  # the threshold of delay

        self.adj = np.zeros((self.car_num, self.car_num))  # adjacency matrix
        self.Car_radius = 100

    def update_fast_fading(self):
        h = 1 / np.sqrt(2) * (np.random.normal(size=(1, 1)) + 1j * np.random.normal(size=(1, 1)))
        fast_fading = np.abs(h) ** 2
        return fast_fading

    def get_path_loss_V2I(self, dist):
        path_loss = 128.1 + 37.6 * np.log10(math.sqrt(dist ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)
        loss = 10 ** (-path_loss / 10)
        return loss

    def step(self, actions):
        self.adj = np.zeros((self.car_num, self.car_num))
        action = np.zeros([self.car_num, 2], dtype='int')
        for i in range(self.car_num):
            action[i][0] = actions[i][0] // self.P_j_level  # the number of the selected RSU
            action[i][1] = actions[i][0] % self.P_j_level  # transmit power of VUE
        done = False
        new_state = np.zeros([self.car_num, self.observation_dim])
        data_bit = np.zeros([self.car_num, ])
        delay = np.zeros([self.car_num, ])
        energy = np.zeros([self.car_num, ])
        penalty = np.zeros([self.car_num, ])
        select_RSU = np.zeros([self.RSU_num, ])
        delay_penalty = np.zeros([self.car_num, ])
        energy_penalty = np.zeros([self.car_num, ])
        for i in range(self.car_num):
            for z in range(self.RSU_num):
                if self.RSU_location[z, 0] == self.state[i, 3 * action[i][0]]:
                    if self.RSU_location[z, 1] == self.state[i, 3 * action[i][0] + 1]:
                        select_RSU[z] += 1

        for i in range(self.car_num):
            input_size = self.state[i, 3 * self.Pre_RSU_number + 2] * 1048576  # kb
            compu_intensity = self.state[i, 3 * self.Pre_RSU_number + 3] * 1000
            delay_up = 0
            delay_down = 0
            dist = np.sqrt((self.Vehicles[i].position[0] - self.state[i, 3 * action[i][0]]) ** 2 +
                           (self.Vehicles[i].position[1] - self.state[i, 3 * action[i][0] + 1]) ** 2)

            tran_size = input_size
            while tran_size > 0:
                if dist > self.Distance_limit_rsu or delay_up > self.delay_thre:
                    penalty[i] = 1
                    if dist > self.Distance_limit_rsu:
                        delay[i] = delay_up
                        energy[i] = (10 ** (self.P_j_car[action[i][1]] / 10 - 3)) * delay_up
                    if delay_up > self.delay_thre:
                        delay_penalty[i] = 1
                        delay[i] = self.delay_thre
                        energy[i] = (10 ** (self.P_j_car[action[i][1]] / 10 - 3)) * self.delay_thre
                    break
                else:
                    if dist == 0:
                        dist = 1e-13
                    pathloss = self.get_path_loss_V2I(dist)
                    channel_gain_up = ((pathloss * self.update_fast_fading()).squeeze()).squeeze()
                    SNR_up = (10 ** (self.P_j_car[action[i][1]] / 10 - 3)) * channel_gain_up / self.sigma2
                    R_up = self.bandwidth * np.log2(1 + SNR_up)
                    tran_size -= R_up * self.slot
                    delay_up += self.slot
                    self.renew_velocity(i)
                    self.renew_position(i)
                    dist = np.sqrt((self.Vehicles[i].position[0] - self.state[i, 3 * action[i][0]]) ** 2 +
                                   (self.Vehicles[i].position[1] - self.state[i, 3 * action[i][0] + 1]) ** 2)
            if penalty[i] == 0:
                energy_up = (10 ** (self.P_j_car[action[i][1]] / 10 - 3)) * delay_up
                if energy_up > self.energy_thre:
                    penalty[i] = 1
                    energy_penalty[i] = 1
                    delay[i] = delay_up
                    energy[i] = (10 ** (self.P_j_car[action[i][1]] / 10 - 3)) * delay_up
                    for _ in range(int((self.delay_thre - delay_up) / self.slot)):
                        self.renew_velocity(i)
                        self.renew_position(i)
                else:
                    g = 0
                    for g in range(self.RSU_num):
                        if self.RSU_location[g, 0] == self.state[i, 3 * action[i][0]]:
                            if self.RSU_location[g, 1] == self.state[i, 3 * action[i][0] + 1]:
                                break
                    delay_exe = round(input_size * compu_intensity / ((self.state[i, 3 * action[i][0] + 2] / 30) / select_RSU[g] * 1e9), 1)
                    energy_exe = self.energy_para * (((self.state[i, 3 * action[i][0] + 2] / 30) / select_RSU[g] * 1e9) ** 2) * compu_intensity * input_size
                    if delay_up + delay_exe > self.delay_thre or energy_up + energy_exe > self.energy_thre:
                        penalty[i] = 1
                        if delay_up + delay_exe > self.delay_thre:
                            delay_penalty[i] = 1
                            delay[i] = self.delay_thre
                            delay_remain = self.delay_thre - delay_up
                            com_size = delay_remain * ((self.state[i, 3 * action[i][0] + 2] / 30) / select_RSU[g] * 1e9) / compu_intensity
                            energy[i] = energy_up + self.energy_para * (((self.state[i, 3 * action[i][0] + 2] / 30) / select_RSU[g] * 1e9) ** 2) * compu_intensity * com_size
                        if energy_up + energy_exe > self.energy_thre:
                            energy_penalty[i] = 1
                            energy[i] = energy_up + energy_exe
                            delay[i] = delay_up + delay_exe
                        for _ in range(int((self.delay_thre - delay_up) / self.slot)):
                            self.renew_velocity(i)
                            self.renew_position(i)
                    else:
                        for _ in range(int(delay_exe / self.slot)):
                            self.renew_velocity(i)
                            self.renew_position(i)
                        dist_down = np.sqrt(
                            (self.Vehicles[i].position[0] - self.state[i, 3 * action[i][0]]) ** 2 +
                            (self.Vehicles[i].position[1] - self.state[i, 3 * action[i][0] + 1]) ** 2)

                        tran_size = input_size * 0.2
                        while tran_size > 0:
                            if dist_down > self.Distance_limit_rsu or delay_up + delay_exe + delay_down > self.delay_thre:
                                penalty[i] = 1
                                if dist_down > self.Distance_limit_rsu:
                                    delay[i] = delay_up + delay_exe + delay_down
                                    energy[i] = energy_up + energy_exe
                                if delay_up + delay_exe + delay_down > self.delay_thre:
                                    delay_penalty[i] = 1
                                    delay[i] = self.delay_thre
                                    energy[i] = energy_up + energy_exe
                                break
                            else:
                                if dist_down == 0:
                                    dist_down = 1e-13
                                pathloss = self.get_path_loss_V2I(dist_down)
                                channel_gain_down = ((pathloss * self.update_fast_fading()).squeeze()).squeeze()
                                SNR_down = (10 ** (self.P_j_rsu / 10 - 3)) * channel_gain_down / self.sigma2
                                R_down = self.bandwidth * np.log2(1 + SNR_down)
                                tran_size -= R_down * self.slot
                                delay_down += self.slot
                                self.renew_velocity(i)
                                self.renew_position(i)
                                dist_down = np.sqrt(
                                    (self.Vehicles[i].position[0] - self.state[i, 3 * action[i][0]]) ** 2 +
                                    (self.Vehicles[i].position[1] - self.state[i, 3 * action[i][0] + 1]) ** 2)
                        if penalty[i] == 0:
                            delay[i] = delay_up + delay_exe + delay_down
                            data_bit[i] = input_size
                            energy[i] = energy_up + energy_exe
                            for _ in range(int((self.delay_thre - delay[i]) / self.slot)):
                                self.renew_velocity(i)
                                self.renew_position(i)
                        else:
                            data_bit[i] = 0
                            if delay_up + delay_exe + delay_down <= self.delay_thre:
                                for _ in range(int((self.delay_thre - delay_up - delay_exe - delay_down) / self.slot)):
                                    self.renew_velocity(i)
                                    self.renew_position(i)
            else:
                if delay_up <= self.delay_thre:
                    for _ in range(int((self.delay_thre - delay_up) / self.slot)):
                        self.renew_velocity(i)
                        self.renew_position(i)

        # Reset computing resources of RSUs
        for t in range(self.RSU_num):
            self.RSU_f_vec[t] = random.randint(2, 4 + self.std_f)

        for i in range(self.car_num):
            # update the positions of the RSUs closest to the VUE
            distance_car_rsu = np.zeros([self.RSU_num, ])
            for j in range(self.RSU_num):
                distance_car_rsu[j] = (self.Vehicles[i].position[0] - self.RSU_location[j, 0]) ** 2 + (
                        self.Vehicles[i].position[1] - self.RSU_location[j, 1]) ** 2
            distance_car_rsu_index = np.argsort(distance_car_rsu)[0:self.Pre_RSU_number]
            # update state
            for k in range(self.Pre_RSU_number):
                rsu_id = distance_car_rsu_index[k]
                new_state[i, 3 * k] = self.RSU_location[rsu_id, 0]
                new_state[i, 3 * k + 1] = self.RSU_location[rsu_id, 1]
                new_state[i, 3 * k + 2] = self.RSU_f_vec[rsu_id] * 30
            new_state[i, 3 * self.Pre_RSU_number] = self.Vehicles[i].position[0]
            new_state[i, 3 * self.Pre_RSU_number + 1] = self.Vehicles[i].position[1]
            new_state[i, 3 * self.Pre_RSU_number + 2] = random.choice(self.input_sizes)  # MB * 2 ** 20
            new_state[i, 3 * self.Pre_RSU_number + 3] = random.choice(self.compute_densitys) / 1000
        # update adjacency matrix
        for i in range(self.car_num):
            for j in range(self.car_num):
                x1 = self.Vehicles[i].position[0]
                y1 = self.Vehicles[i].position[1]
                x2 = self.Vehicles[j].position[0]
                y2 = self.Vehicles[j].position[1]
                dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dis < self.Car_radius and dis != 0:
                    self.adj[i, j] = 1
        # Computing rewards, average delay per user, energy consumption per user, and effective computing bits per user
        self.state = new_state
        self.old_action = action
        scalar_para = 2
        success = self.car_num
        delay_success = self.car_num
        energy_success = self.car_num
        for w in range(self.car_num):
            if energy[w] > self.energy_thre or delay[w] > self.delay_thre:
                penalty[w] = 1
                if delay[w] > self.delay_thre:
                    delay_penalty[w] = 1
                if energy[w] > self.energy_thre:
                    energy_penalty[w] = 1
            if penalty[w] == 1:
                success -= 1
            if delay_penalty[w] == 1:
                delay_success -= 1
            if energy_penalty[w] == 1:
                energy_success -= 1
        Data_bit = sum(data_bit)
        Delay = sum(delay)
        Energy = sum(energy)
        if success == 0:
            reward = -2 * self.car_num
        else:
            reward = (Data_bit / self.data_max) / (Energy / (self.energy_thre * scalar_para)) - 2 * (self.car_num - success)

        reward = reward / self.car_num
        Data_bit = Data_bit / self.car_num  # Mbit
        Delay = Delay / self.car_num
        Energy = Energy / self.car_num
        return self.state, self.state, reward, Data_bit / 1048576, Energy, success / self.car_num, done, self.adj, Delay, delay_success / self.car_num, energy_success / self.car_num

    def reset(self):
        # Initialize the VUE's position and adjacency matrix
        self.adj = np.zeros((self.car_num, self.car_num))
        self.state = np.zeros([self.car_num, self.observation_dim])
        new_state = np.zeros([self.car_num, self.observation_dim])
        for j in range(int(self.car_num / 4)):
            ind = np.random.randint(0, len(self.down_lanes))
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'd'
            self.Vehicles.append(
                Vehicle(start_position, start_direction, np.random.randint(self.vel_m - 5, self.vel_m + 5)))

            start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'u'
            self.Vehicles.append(
                Vehicle(start_position, start_direction, np.random.randint(self.vel_m - 5, self.vel_m + 5)))

            start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.Vehicles.append(
                Vehicle(start_position, start_direction, np.random.randint(self.vel_m - 5, self.vel_m + 5)))

            start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.Vehicles.append(
                Vehicle(start_position, start_direction, np.random.randint(self.vel_m - 5, self.vel_m + 5)))
        for i in range(self.car_num):
            new_state[i, 3 * self.Pre_RSU_number] = self.Vehicles[i].position[0]
            new_state[i, 3 * self.Pre_RSU_number + 1] = self.Vehicles[i].position[1]

        for i in range(self.car_num):
            for j in range(self.car_num):
                x1 = self.Vehicles[i].position[0]
                y1 = self.Vehicles[i].position[1]
                x2 = self.Vehicles[j].position[0]
                y2 = self.Vehicles[j].position[0]
                dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dis < self.Car_radius and dis != 0:
                    self.adj[i, j] = 1

        # update new state
        distance_car_rsu = np.zeros([self.RSU_num, ])
        for i in range(self.car_num):
            for j in range(self.RSU_num):
                distance_car_rsu[j] = ((self.Vehicles[i].position[0] - self.RSU_location[j, 0]) ** 2
                                       + (self.Vehicles[i].position[1] - self.RSU_location[j, 1]) ** 2)
            distance_car_rsu_index = np.argsort(distance_car_rsu)[0:self.Pre_RSU_number]
            for k in range(self.Pre_RSU_number):
                rsu_id = distance_car_rsu_index[k]
                new_state[i, 3 * k] = self.RSU_location[rsu_id, 0]
                new_state[i, 3 * k + 1] = self.RSU_location[rsu_id, 1]
                new_state[i, 3 * k + 2] = self.RSU_f_vec[rsu_id] * 30
            new_state[i, 3 * self.Pre_RSU_number + 2] = random.choice(self.input_sizes)  # MB * 2 ** 20
            new_state[i, 3 * self.Pre_RSU_number + 3] = random.choice(self.compute_densitys) / 1000

        self.state = new_state
        return self.state, self.state, self.adj

    def renew_position(self, i):
        # ===============
        # This function updates the position of each vehicle
        # ===============

        delta_distance = self.Vehicles[i].velocity * self.slot
        change_direction = False
        if self.Vehicles[i].direction == 'u':
            for j in range(len(self.left_lanes)):
                if (self.Vehicles[i].position[1] <= self.left_lanes[j]) and (
                        (self.Vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                    if np.random.uniform(0, 1) < 0.4:
                        self.Vehicles[i].position = [self.Vehicles[i].position[0] - (
                                delta_distance - (self.left_lanes[j] - self.Vehicles[i].position[1])),
                                                     self.left_lanes[j]]
                        self.Vehicles[i].direction = 'l'
                        change_direction = True
                        break
            if change_direction == False:
                for j in range(len(self.right_lanes)):
                    if (self.Vehicles[i].position[1] <= self.right_lanes[j]) and (
                            (self.Vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                        if np.random.uniform(0, 1) < 0.4:
                            self.Vehicles[i].position = [self.Vehicles[i].position[0] + (
                                    delta_distance + (self.right_lanes[j] - self.Vehicles[i].position[1])),
                                                         self.right_lanes[j]]
                            self.Vehicles[i].direction = 'r'
                            change_direction = True
                            break
            if change_direction == False:
                self.Vehicles[i].position[1] += delta_distance
        if (self.Vehicles[i].direction == 'd') and (change_direction == False):
            for j in range(len(self.left_lanes)):
                if (self.Vehicles[i].position[1] >= self.left_lanes[j]) and (
                        (self.Vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                    if (np.random.uniform(0, 1) < 0.4):
                        self.Vehicles[i].position = [self.Vehicles[i].position[0] - (
                                delta_distance - (self.Vehicles[i].position[1] - self.left_lanes[j])),
                                                     self.left_lanes[j]]
                        self.Vehicles[i].direction = 'l'
                        change_direction = True
                        break
            if change_direction == False:
                for j in range(len(self.right_lanes)):
                    if (self.Vehicles[i].position[1] >= self.right_lanes[j]) and (
                            self.Vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                        if (np.random.uniform(0, 1) < 0.4):
                            self.Vehicles[i].position = [self.Vehicles[i].position[0] + (
                                    delta_distance + (self.Vehicles[i].position[1] - self.right_lanes[j])),
                                                         self.right_lanes[j]]
                            self.Vehicles[i].direction = 'r'
                            change_direction = True
                            break
            if change_direction == False:
                self.Vehicles[i].position[1] -= delta_distance
        if (self.Vehicles[i].direction == 'r') and (change_direction == False):
            for j in range(len(self.up_lanes)):
                if (self.Vehicles[i].position[0] <= self.up_lanes[j]) and (
                        (self.Vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):
                    if (np.random.uniform(0, 1) < 0.4):
                        self.Vehicles[i].position = [self.up_lanes[j], self.Vehicles[i].position[1] + (
                                delta_distance - (self.up_lanes[j] - self.Vehicles[i].position[0]))]
                        change_direction = True
                        self.Vehicles[i].direction = 'u'
                        break
            if change_direction == False:
                for j in range(len(self.down_lanes)):
                    if (self.Vehicles[i].position[0] <= self.down_lanes[j]) and (
                            (self.Vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                        if (np.random.uniform(0, 1) < 0.4):
                            self.Vehicles[i].position = [self.down_lanes[j], self.Vehicles[i].position[1] - (
                                    delta_distance - (self.down_lanes[j] - self.Vehicles[i].position[0]))]
                            change_direction = True
                            self.Vehicles[i].direction = 'd'
                            break
            if change_direction == False:
                self.Vehicles[i].position[0] += delta_distance
        if (self.Vehicles[i].direction == 'l') and (change_direction == False):
            for j in range(len(self.up_lanes)):

                if (self.Vehicles[i].position[0] >= self.up_lanes[j]) and (
                        (self.Vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):
                    if (np.random.uniform(0, 1) < 0.4):
                        self.Vehicles[i].position = [self.up_lanes[j], self.Vehicles[i].position[1] + (
                                delta_distance - (self.Vehicles[i].position[0] - self.up_lanes[j]))]
                        change_direction = True
                        self.Vehicles[i].direction = 'u'
                        break
            if change_direction == False:
                for j in range(len(self.down_lanes)):
                    if (self.Vehicles[i].position[0] >= self.down_lanes[j]) and (
                            (self.Vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                        if (np.random.uniform(0, 1) < 0.4):
                            self.Vehicles[i].position = [self.down_lanes[j], self.Vehicles[i].position[1] - (
                                    delta_distance - (self.Vehicles[i].position[0] - self.down_lanes[j]))]
                            change_direction = True
                            self.Vehicles[i].direction = 'd'
                            break
                if change_direction == False:
                    self.Vehicles[i].position[0] -= delta_distance

        # if it comes to an exit
        if (self.Vehicles[i].position[0] < 0) or (self.Vehicles[i].position[1] < 0) or (
                self.Vehicles[i].position[0] > self.width) or (self.Vehicles[i].position[1] > self.height):
            if (self.Vehicles[i].direction == 'u'):
                self.Vehicles[i].direction = 'r'
                self.Vehicles[i].position = [self.Vehicles[i].position[0], self.right_lanes[-1]]
            else:
                if (self.Vehicles[i].direction == 'd'):
                    self.Vehicles[i].direction = 'l'
                    self.Vehicles[i].position = [self.Vehicles[i].position[0], self.left_lanes[0]]
                else:
                    if (self.Vehicles[i].direction == 'l'):
                        self.Vehicles[i].direction = 'u'
                        self.Vehicles[i].position = [self.up_lanes[0], self.Vehicles[i].position[1]]
                    else:
                        if (self.Vehicles[i].direction == 'r'):
                            self.Vehicles[i].direction = 'd'
                            self.Vehicles[i].position = [self.down_lanes[-1], self.Vehicles[i].position[1]]

    def renew_velocity(self, i):
        new_velocity = self.gaussian_mobility(self.Vehicles[i].velocity)
        self.Vehicles[i].velocity = new_velocity

    def gaussian_mobility(self, velocity_old):
        alpha = 0.1
        w = np.random.normal(0, 1)
        velocity_mean = random.randint(self.vel_m - 5, self.vel_m + 5)
        velocity_new = velocity_old * alpha + velocity_mean * (1 - alpha) + 0.1 * np.sqrt(1 - alpha ** 2) * w
        return velocity_new
