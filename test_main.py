import gym
import torch
import gym_delta_robot_trampoline
import time
import numpy as np
from gym_delta_robot_trampoline.envs.delta_robot_env import DeltaRobotEnv

def main():
    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                             torch.nn.Linear(64, 2))
    # agent = TRPOAgent(policy=nn)
    #
    # agent.load_model("agent.pth")
    # agent.train("SimpleDriving-v0", seed=0, batch_size=5000, iterations=100,
    #             max_episode_length=250, verbose=True)
    # agent.save_model("agent.pth")
    env = DeltaRobotEnv()
    ob = env.reset()
    while True:
        randoms = np.random.rand(3)
        randoms = 100*randoms - 50
        action = list(randoms)
        ob, _, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1/30)


if __name__ == '__main__':
    main()
