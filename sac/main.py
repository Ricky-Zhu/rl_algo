from sac import SACAgent
from buffer import ReplayBuffer
import numpy as np
import torch


def train_loop(env, test_env, args):
    agent = SACAgent(env, test_env, args)
    buffer = ReplayBuffer(env.observation_space.shape[0],
                          env.action_space.shape[0],
                          args.buffer_size,
                          args.device)
    total_interaction_steps = 0

    # initialize the buffer
    o = env.reset()
    for i in range(args.initialize_buffer_steps):
        a = env.action_space.sample()
        o2, r, d, _ = env.step(a)
        buffer.store(o, a, r, o2, d)
        if d:
            env.reset()

    for epoch in range(args.total_epochs):
        for ep in range(args.episodes_per_epoch):
            o = env.reset()

            for i in range(args.max_episode_length):
                a = agent.get_action(o)
                o2, r, d, _ = env.step(a)
                total_interaction_steps += 1

                buffer.store(o, a, r, o2, d)
                o = o2
                if d:
                    o = env.reset()
                data = buffer.sample_batch(batch_size=args.batch_size)
                agent.update(data)

        if (epoch + 1) % args.evaluate_interval_epochs == 0:
            average_return = 0
            for i in range(args.evaluation_nums):
                o = test_env.reset()
                d = False
                while not d:
                    a = agent.get_action(o, deterministic=True)
                    o2, r, d, _ = test_env.step(a)
                    average_return += r
                    o = o2
            average_return /= args.evaluation_nums
            print('epoch:{}/{}, total interaction steps:{}, average return:{}'.format(epoch, args.total_epochs,
                                                                                      total_interaction_steps,
                                                                                      average_return))


if __name__ == "__main__":
    import argparse
    import gym

    parser = argparse.ArgumentParser()

    # env parameters
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')

    # agent parameters
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--polyak', type=float, default=0.005)
    parser.add_argument('--q-lr', type=float, default=3e-4)
    parser.add_argument('--p-lr', type=float, default=3e-4)
    parser.add_argument('--reward-scale', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')

    # buffer parameters
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=int(1e6))

    # training parameters
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--seed', '-s', type=int, default=44)
    parser.add_argument('--total-epochs', type=int, default=300)
    parser.add_argument('--episodes-per-epoch', type=int, default=1)
    parser.add_argument('--initialize-buffer-steps', type=int, default=10000)
    parser.add_argument('--max-episode-length', type=int, default=1000)
    parser.add_argument('--update-cycles', type=int, default=1000)
    parser.add_argument('--evaluate-interval-epochs', type=int, default=5)
    parser.add_argument('--evaluation-nums', type=int, default=5)
    args = parser.parse_args()

    ##########################################################################
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)  # to ensure during the early random exploration the data the same

    test_env = gym.make(args.env)
    test_env.seed(args.seed + 100)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loop(env, test_env, args)
