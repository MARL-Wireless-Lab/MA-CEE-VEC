import time
import numpy as np
import torch
from base_runner import Runner
from tqdm import tqdm


def _t2n(x):
    return x.detach().cpu().numpy()


class VARunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(VARunner, self).__init__(config)
        self.episode_rewards = []
        self.episode_datas = []
        self.episode_energys = []
        self.episode_delays = []
        self.episode_delay_successs = []
        self.episode_energy_successs = []
        self.episode_successs = []
        self.valueloss = []
        self.policyloss = []

    def run(self, vari):

        start = time.time()
        episodes = 2000

        for episode in tqdm(range(episodes)):
            self.warmup()
            episode_rewards = 0
            episode_data = 0
            episode_energys = 0
            episode_delay = 0
            episode_delay_successs = 0
            episode_energy_successs = 0
            episode_successs = 0
            episode_dones = 0
            # if self.use_linear_lr_decay:
            #     self.trainer.policy.lr_decay(episode, episodes)

            step = 0
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, data, energy, success, dones, delay, delay_success, energy_success = self.envs.step(actions)
                episode_rewards += rewards
                episode_data += data
                episode_energys += energy
                episode_delay += delay
                episode_delay_successs += delay_success
                episode_energy_successs += energy_success
                episode_successs += success
                episode_dones += dones

                data = obs, share_obs, rewards, dones, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.episode_rewards.append(episode_rewards)
            self.episode_datas.append(episode_data / self.episode_length)
            self.episode_energys.append(episode_energys / self.episode_length)
            self.episode_delays.append(episode_delay / self.episode_length)
            self.episode_delay_successs.append(episode_delay_successs / self.episode_length)
            self.episode_energy_successs.append(episode_energy_successs / self.episode_length)
            self.episode_successs.append(episode_successs / self.episode_length)
            what = 'car_num={}'.format(vari)
            np.save('./result/MAPPO' + '/{}'.format(what), self.episode_rewards)
            np.save('./result/MAPPO' + '/{}_data'.format(what), self.episode_datas)
            np.save('./result/MAPPO' + '/{}_energy'.format(what), self.episode_energys)
            np.save('./result/MAPPO' + '/{}_delay'.format(what), self.episode_delays)
            np.save('./result/MAPPO' + '/{}_delay_success'.format(what), self.episode_delay_successs)
            np.save('./result/MAPPO' + '/{}_energy_success'.format(what), self.episode_energy_successs)
            np.save('./result/MAPPO' + '/{}_success'.format(what), self.episode_successs)
            self.compute()
            train_infos = self.train()
            self.valueloss.append(train_infos['value_loss'])
            self.policyloss.append(train_infos['policy_loss'])
            # np.save('./result/MAPPO' + '/{}_valueLoss'.format(what), self.valueloss)
            # np.save('./result/MAPPO' + '/{}_policyLoss'.format(what), self.policyloss)
            # print(episode, 'Value loss: ', train_infos['value_loss'], 'Policy loss: ', train_infos['policy_loss'])
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

    def warmup(self):
        # reset env
        obs, share_obs = self.envs.reset()

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()


    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(self.buffer.share_obs[step],
                                            self.buffer.obs[step],
                                            self.buffer.actions_onehot[step],
                                            self.buffer.rnn_states[step],
                                            self.buffer.rnn_states_critic[step])
        # [self.envs, agents, dim]
        values = np.array(_t2n(value))
        actions = np.array(_t2n(action))
        action_log_probs = np.array(_t2n(action_log_prob))
        rnn_states = np.array(_t2n(rnn_state))
        rnn_states_critic = np.array(_t2n(rnn_state_critic))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        if dones:
            rnn_states = np.zeros((self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            rnn_states_critic = np.zeros((self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards)