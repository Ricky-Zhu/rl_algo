import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import json
import gym
import argparse
from agent import PPO


################################### Training ###################################
def train():
    print("============================================================================================")
    parser = argparse.ArgumentParser(description='PPO')
    parser.add_argument('--env_name', default="Hopper-v2", type=str)
    parser.add_argument('--has_continuous_action_space', action="store_false")
    parser.add_argument('--max_ep_len', default=1000, type=int)
    parser.add_argument('--max_training_timesteps', default=int(1e6), type=int)
    parser.add_argument('--print_freq', default=10000, type=int)
    parser.add_argument('--log_freq', default=2000, type=int)
    parser.add_argument('--action_std', default=0.6, type=float)
    parser.add_argument('--action_std_decay_rate', default=0.05, type=float)
    parser.add_argument('--action_std_decay_freq', default=int(2.5e5), type=int)
    parser.add_argument('--min_action_std', default=0.1, type=float)
    parser.add_argument('--update_timestep', default=4000, type=int)
    parser.add_argument('--K_epochs', default=80, type=int)
    parser.add_argument('--eps_clip', default=0.2, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lr_actor', default=0.0003, type=float)
    parser.add_argument('--lr_critic', default=0.001, type=float)
    parser.add_argument('--total_test_episodes', default=5, type=int)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()

    ####### initialize environment hyperparameters ######
    env_name = args.env_name

    has_continuous_action_space = args.has_continuous_action_space  # continuous action space; else discrete

    max_ep_len = args.max_ep_len  # max timesteps in one episode
    max_training_timesteps = args.max_training_timesteps  # break training loop if timeteps > max_training_timesteps

    print_freq = args.print_freq  # print avg reward in the interval (in num timesteps)
    log_freq = args.log_freq  # log avg reward in the interval (in num timesteps)

    action_std = args.action_std  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = args.action_std_decay_rate  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = args.min_action_std  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = args.action_std_decay_freq  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = args.update_timestep  # update policy every n timesteps
    K_epochs = args.K_epochs  # update policy for K epochs in one PPO update

    eps_clip = args.eps_clip  # clip parameter for PPO
    gamma = args.gamma  # discount factor

    lr_actor = args.lr_actor  # learning rate for actor network
    lr_critic = args.lr_critic  # learning rate for critic network

    random_seed = args.random_seed  # set random seed if required (0 = no random seed)
    device = args.device
    total_test_episodes = args.total_test_episodes
    #####################################################

    print("training environment name : " + env_name)

    env = gym.make(env_name)
    env.seed(random_seed)
    test_env = gym.make(env_name)
    test_env.seed(random_seed + 100)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    current_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = log_dir + '/' + env_name + '/' + current_time + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### create new log file for each run
    log_f_name = log_dir + 'PPO_' + env_name + "_log" + ".csv"
    print("logging at : " + log_f_name)

    # save all the hyper-parameters
    with open(log_dir + "/parameters.json", 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)

    #####################################################

    checkpoint_path = log_dir + "PPO_{}.pth".format(env_name)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    device, action_std, )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print(
                    "Episode : {} \t\t Timestep : {} \t\t Running Average Reward : {}".format(
                        i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

                # save model weights
                ppo_agent.save(checkpoint_path)

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
