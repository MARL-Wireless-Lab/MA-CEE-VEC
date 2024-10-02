import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from Env_mhd_energy import AUTOVANET_CV1
from tqdm import tqdm

# thedevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Net(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )


    def forward(self, x):
        return self.layers(x)


class DDQN(nn.Module):
    def __init__(self,
                 n_state,
                 n_action,
                 learning_rate=0.0005,  # 0.0005
                 gamma=0.9,  # 0.9
                 epsilon=0.95,  # 0.95
                 replace_target_iter=100,  # 100
                 memory_size=800,
                 batch_size=32,
                 ):
        super(DDQN, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.memory_size = memory_size
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.eval_net = Net(self.n_state, self.n_action)
        self.target_net = Net(self.n_state, self.n_action)

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((memory_size, self.n_state * 2 + 2))  # s, a, r, s_, t
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

        self.verbose = 0

        self.lossHistory = []
        #self.to(thedevice)

    def choose_action(self, s, epsilon):
        x = torch.unsqueeze(torch.FloatTensor(s), 0)
        # input only one sample
        if np.random.uniform() < epsilon:  # greedy
            action = np.random.randint(0, self.n_action)
        else:  # random
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            if self.verbose >= 1:
                print('Target DQN params replaced')
        self.learn_step_counter += 1

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:  # 如果经验池满了
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]
        batch_state = batch_memory[:, :self.n_state]
        batch_action = batch_memory[:, self.n_state].astype(int)  # float -> int
        batch_reward = batch_memory[:, self.n_state + 1]
        batch_state_ = batch_memory[:, -self.n_state:]

        #CPU
        b_s = Variable(torch.FloatTensor(np.float32(batch_state)))
        b_s_ = Variable(torch.FloatTensor(np.float32(batch_state_)))


        q_eval = self.eval_net(b_s)
        q_next = self.target_net(b_s_)
        q_eval4next = self.eval_net(b_s_)
        q_eval4next = q_eval4next.detach().numpy()

        q_target = q_eval.clone()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        max_act4next = np.argmax(q_eval4next, axis=1)
        max_postq = q_next[batch_index, max_act4next]

        for i, action in enumerate(batch_action):
            q_target[i, action] = batch_reward[i] + self.gamma * max_postq[i]

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


# ################## SETTINGS ######################
IS_TRAIN = 1
IS_TEST = 1 - IS_TRAIN

setup_seed(2023)

car_num = [20]
for f in range(len(car_num)):
    env = AUTOVANET_CV1(car_num[f])

    n_episode = 2000
    n_step_per_episode = 100

    epsi_final = 0.1  # 0.1
    epsi_anneal_length = 500  # 500
    total_train_steps = 0

    train_frequency = 5
    # ------------------------ initialize ----------------------------
    num_input = env.state_dim
    num_output = env.action_dim
    n_agent = env.car_num

    global_agent = Net(num_input, num_output)
    agents = []
    for ind_agent in range(n_agent):  # initialize agents
        print("Initializing agent", ind_agent)
        agent = DDQN(num_input, num_output)
        agents.append(agent)
    # ------------------------- Training -----------------------------
    record_loss = []
    ep_rewards = []
    ep_data = []
    ep_energy = []
    ep_delay = []
    ep_delay_success = []
    ep_energy_success = []
    ep_success = []
    save_path = './result' + '/' + 'IDDQN'
    if IS_TRAIN:
        for i_episode in tqdm(range(n_episode)):
            # print("-------------------------")
            # print('Episode:', i_episode)

            if i_episode < epsi_anneal_length:
                epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
            else:
                epsi = epsi_final

            env.reset()
            episode_reward = 0
            episode_data = 0
            episode_energy = 0
            episode_delay = 0
            episode_delay_success = 0
            episode_energy_success = 0
            episode_success = 0
            i_step = 0
            for i_step in range(n_step_per_episode):
                time_step = i_episode * n_step_per_episode + i_step
                state_old_all = []
                action_all = []
                for i in range(n_agent):
                    state = env.state[i]
                    state_old_all.append(state)
                    action_temp = agents[i].choose_action(state, epsi)
                    action_all.append(action_temp)

                action_temp = action_all.copy()
                state_new, train_reward, data, energy, success, done, delay, delay_success, energy_success = env.step(action_temp)
                episode_reward += train_reward
                episode_data += data
                episode_energy += energy
                episode_delay += delay
                episode_delay_success += delay_success
                episode_energy_success += energy_success
                episode_success += success
                total_train_steps += 1

                for i in range(n_agent):
                    state_old = state_old_all[i]
                    action = action_all[i]
                    state_n = state_new[i]
                    agents[i].store_transition(state_old, action, train_reward, state_n)

                    # training this agent
                    if agents[i].memory_counter > agents[i].batch_size and total_train_steps % train_frequency == 0:
                        loss_val_batch = agents[i].learn().item()
                        record_loss.append(loss_val_batch)
            #ep_rewards.append(episode_reward)
                if done:
                    break
            ep_rewards.append(episode_reward)
            ep_data.append(episode_data / i_step)
            ep_energy.append(episode_energy / i_step)
            ep_delay.append(episode_delay / i_step)
            ep_delay_success.append(episode_delay_success / i_step)
            ep_energy_success.append(episode_energy_success / i_step)
            ep_success.append(episode_success / i_step)
            what = 'car_num={}'.format(car_num[f])
            np.save('./result/IDDQN' + '/{}'.format(what), ep_rewards)
            np.save('./result/IDDQN' + '/{}_data'.format(what), ep_data)
            np.save('./result/IDDQN' + '/{}_energy'.format(what), ep_energy)
            np.save('./result/IDDQN' + '/{}_delay'.format(what), ep_delay)
            np.save('./result/IDDQN' + '/{}_delay_success'.format(what), ep_delay_success)
            np.save('./result/IDDQN' + '/{}_energy_success'.format(what), ep_energy_success)
            np.save('./result/IDDQN' + '/{}_success'.format(what), ep_success)


