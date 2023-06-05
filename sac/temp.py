import torch
import torch.nn as nn
import numpy as np
from buffer import ReplayBuffer
from models import MLPActorCritic
from copy import deepcopy
import itertools
import time
import random

def sac(env, test_env, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=3e-4, alpha=0.2, batch_size=256, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        save_freq=1, reward_scale=5, device='cpu'):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac).to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = reward_scale * r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    def compute_pi_loss(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        a_new, a_new_logprobs = ac.pi(o)

        # # update the alpha
        # alpha_loss = -(log_alpha * (a_new_logprobs + target_entropy).detach()).mean()
        # alpha_optim.zero_grad()
        # alpha_loss.backward()
        # alpha_optim.step()
        #
        # alpha = log_alpha.exp()

        # Target Q-values
        q1_pi = ac.q1(o, a_new)
        q2_pi = ac.q2(o, a_new)
        q_pi = torch.min(q1_pi, q2_pi)

        backup = alpha * a_new_logprobs - q_pi
        loss_pi = backup.mean()
        pi_info = dict(LogPi=loss_pi.detach().cpu().numpy())
        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = torch.optim.Adam(q_params, lr=lr)

    # set up the alpha optimizer
    # entropy target
    # target_entropy = -np.prod(env.action_space.shape).item()
    # log_alpha = torch.zeros(1, requires_grad=True, device=device)
    # alpha_optim = torch.optim.Adam([log_alpha], lr=lr)

    def update(data):
        logger = dict()

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        logger['LossQ'] = loss_q.item()

        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_pi_loss(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        logger['LossPi'] = loss_pi.item()

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32, device=device),
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
        return ep_ret / float(num_test_episodes)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

                # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs):
            #     logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            avg_ret = test_agent()
            print('steps:{}/{}, epoch:{}/{}, average return:{}'.format(t, total_steps, epoch, epochs, avg_ret))

    end_time = time.time()
    duration = end_time - start_time
    print('duration:{}'.format(duration))


if __name__ == '__main__':
    import argparse
    import gym

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=44)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--steps-per-epoch', type=int, default=1000)
    parser.add_argument('--reward-scale', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # torch.set_num_threads(torch.get_num_threads())

    # construct env and test env
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)  # to ensure during the early random exploration the data the same

    test_env = gym.make(args.env)
    test_env.seed(args.seed + 100)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    sac(env=env, test_env=test_env, actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=args.hid),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch, reward_scale=args.reward_scale, device=args.device
        )
