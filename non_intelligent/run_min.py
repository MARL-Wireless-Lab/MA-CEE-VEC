import argparse
import os
import random
import numpy as np
import torch
from Env_mhd_energy import AUTOVANET_CV1
from tqdm import tqdm
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)
"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env', help='environment ID', default='AUTOVANET-C-v2')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    args = parser.parse_args()
    return args


# arguments of vdn、 qmix、 qtran
def get_random_args(args):
    args.episode_limit = 100
    # the number of the epoch to train the agent
    args.n_epoch = 500
    # # how often to evaluate
    args.evaluate_cycle = 20
    return args

class Runner:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.episode_rewards = []
        self.episode_datas = []
        self.episode_energys = []
        self.episode_delays = []
        self.episode_delay_success = []
        self.episode_energy_success = []
        self.episode_success = []

        self.save_path = self.args.result_dir + '/' + 'min'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, vari):
        for epoch in tqdm(range(self.args.n_epoch)):
            episode_reward, episode_data, episode_energy, episode_delay, episode_success, episode_delay_success, episode_energy_success = self.generate_episode()
            self.episode_rewards.append(episode_reward)
            self.episode_datas.append(episode_data)
            self.episode_energys.append(episode_energy)
            self.episode_delays.append(episode_delay)
            self.episode_delay_success.append(episode_delay_success)
            self.episode_energy_success.append(episode_energy_success)
            self.episode_success.append(episode_success)
            what = 'car_num={}'.format(vari)
            np.save(self.save_path + '/{}'.format(what), self.episode_rewards)
            np.save(self.save_path + '/{}_data'.format(what), self.episode_datas)
            np.save(self.save_path + '/{}_energy'.format(what), self.episode_energys)
            np.save(self.save_path + '/{}_delay'.format(what), self.episode_delays)
            np.save(self.save_path + '/{}_delay_success'.format(what), self.episode_delay_success)
            np.save(self.save_path + '/{}_energy_success'.format(what), self.episode_energy_success)
            np.save(self.save_path + '/{}_success'.format(what), self.episode_success)


    def generate_episode(self):
        obs = self.env.reset()
        terminated = False
        step = 0
        episode_reward = 0  # cumulative rewards
        episode_data = 0
        episode_energy = 0
        episode_delay = 0
        episode_delay_success = 0
        episode_energy_success = 0
        episode_success = 0
        while not terminated and step < self.args.episode_limit:
            obs, reward, data, energy, success, done, delay, delay_success, energy_success = self.env.step()
            episode_reward += reward
            episode_data += data
            episode_energy += energy
            episode_delay += delay
            episode_delay_success += delay_success
            episode_energy_success += energy_success
            episode_success += success
            step += 1

        return episode_reward, episode_data / step, episode_energy / step, episode_delay / step, episode_success / step, episode_delay_success / step, episode_energy_success / step


if __name__ == '__main__':
    car_num = [20]
    for i in range(len(car_num)):
        args = get_common_args()
        args = get_random_args(args)
        env = AUTOVANET_CV1(car_num[i])
        # args.n_actions = 2 ** env.action_dim
        args.action_dim = env.action_dim
        args.n_agents = env.car_num
        # args.state_shape = env.state_dim
        args.obs_shape = env.state_dim
        runner = Runner(env, args)
        runner.run(car_num[i])

