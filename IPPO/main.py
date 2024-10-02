from datetime import datetime
import torch
import numpy as np
import random
from tqdm import tqdm
from Env_mhd_energy import AUTOVANET_CV1
from PPO_Agent import PPO
import copy


################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    has_continuous_action_space = False  # continuous action space; else discrete
    max_ep_len = 100  # max timesteps in one episode
    max_training_timesteps = 2000  # break training loop if timeteps > max_training_timesteps
    save_model_freq = 50  # save model frequency (in num timesteps)

    ################ PPO hyperparameters ################
    update_timestep = 1  # 1 update policy every n timesteps
    K_epochs = 50  # 50 update policy for K epochs in one PPO update
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.85  # discount factor 0.85
    lr_actor = 1e-4  # learning rate for actor network
    lr_critic = 1e-4  # learning rate for critic network
    random_seed = 0  # set random seed if required (0 = no random seed)

    save_path = './result' + '/IPPO'

    # training loop
    def run(vari):
        def merge_fl(agents):
            net_params = []
            for i in range(len(agents)):
                net_params.append(agents[i].policy.actor.state_dict())
            avg_params = copy.deepcopy(net_params[0])
            for key in avg_params.keys():
                for i in range(1, len(net_params)):
                    avg_params[key] += net_params[i][key]
                avg_params[key] = torch.div(avg_params[key], len(net_params))
            for i in range(len(agents)):
                agents[i].policy.actor.load_state_dict(avg_params)


        env = AUTOVANET_CV1(vari)

        # state space dimension
        state_dim = env.state_dim

        # action space dimension
        if has_continuous_action_space:
            action_dim = env.action_space.shape[0]
        else:
            action_dim = env.action_dim

        if random_seed:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", random_seed)
            torch.manual_seed(random_seed)
            env.seed(random_seed)
            np.random.seed(random_seed)

        # initialize a PPO agent
        agents = []
        for i in range(env.car_num):
            ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space)
            agents.append(ppo_agent)
        i_episode = 0
        rewards = []
        datas = []
        energys = []
        delays = []
        delay_successs = []
        energy_successs = []
        successs = []
        for i_episode in tqdm(range(max_training_timesteps)):
        # while i_episode <= max_training_timesteps:
            state = env.reset()
            ep_reward = 0
            ep_data = 0
            ep_energy = 0
            ep_delay = 0
            ep_delay_success = 0
            ep_energy_success = 0
            ep_success = 0
            for t in range(max_ep_len):
                # select action with policy
                actions = []
                for i in range(len(agents)):
                    action = agents[i].select_action(state[i])
                    actions.append(action)
                state, reward, data, energy, success, done, delay, delay_success, energy_success = env.step(actions)

                # saving reward and is_terminals
                for i in range(len(agents)):
                    agents[i].buffer.rewards.append(reward)
                    agents[i].buffer.is_terminals.append(done)

                ep_reward += reward
                ep_data += data
                ep_energy += energy
                ep_delay += delay
                ep_delay_success += delay_success
                ep_energy_success += energy_success
                ep_success += success

            rewards.append(ep_reward)
            datas.append(ep_data / max_ep_len)
            energys.append(ep_energy / max_ep_len)
            delays.append(ep_delay / max_ep_len)
            delay_successs.append(ep_delay_success / max_ep_len)
            energy_successs.append(ep_energy_success / max_ep_len)
            successs.append(ep_success / max_ep_len)
            what = 'car_num={}'.format(vari)
            np.save(save_path + '/{}'.format(what), rewards)
            np.save(save_path + '/{}_data'.format(what), datas)
            np.save(save_path + '/{}_energy'.format(what), energys)
            np.save(save_path + '/{}_delay'.format(what), delays)
            np.save(save_path + '/{}_delay_success'.format(what), delay_successs)
            np.save(save_path + '/{}_energy_success'.format(what), energy_successs)
            np.save(save_path + '/{}_success'.format(what), successs)

            # update PPO agent
            if i_episode % update_timestep == 0:
                for i in range(len(agents)):
                    agents[i].update()

            if i_episode % 20 == 0:
                merge_fl(agents)

            i_episode += 1
            if i_episode % save_model_freq == 0:
                for i in range(len(agents)):
                    agents[i].save()

    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)

    car_num = [20]
    for i in range(len(car_num)):
        run(car_num[i])


if __name__ == '__main__':
    train()
