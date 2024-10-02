import numpy as np
import random
import torch
from config import get_config
from runner import VARunner as Runner
from Env_mhd_energy import AUTOVANET_CV1


if __name__ == "__main__":
    all_args = get_config()

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(all_args.n_training_threads)

    # seed
    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)

    car_num = [20]
    for i in range(len(car_num)):
        envs = AUTOVANET_CV1(car_num[i])
        num_agents = envs.car_num

        config = {
            "all_args": all_args,
            "envs": envs,
            "num_agents": num_agents,
            "device": device,
        }

        runner = Runner(config)
        runner.run(car_num[i])
