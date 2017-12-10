#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_cont import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.policies import CnnPolicy
from baselines.acktr.lynxEnv import Lynx
from baselines.acktr.value_functions import NeuralNetValueFunction
import tensorflow as tf
import argparse
from baselines.acktr.MLP import MlpPolicy
MLP = False
def train(env_id, num_timesteps, seed):
    env=Lynx()
    #env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(1)))
    set_global_seeds(seed)
    #env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    with tf.Session(config=tf.ConfigProto()) as sess:
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            if MLP:
                policy = MlpPolicy(sess, ob_space=env.observation_space, ac_space=env.action_space)
            else:
                policy = GaussianMlpPolicy(ob_dim,ac_dim)

        learn(env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=50,
            desired_kl=0.002,
            num_timesteps=num_timesteps, animate=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mujoco benchmark.')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', help='environment ID', type=str, default="Reacher-v1")
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

