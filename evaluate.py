from environment import MysteryControlEnv
from agent import MySmartAgent
import numpy as np
import csv

EPISODES = 50

env = MysteryControlEnv()

agent = MySmartAgent(env.action_space, env.observation_space)

scores = []

for episode in range(EPISODES):

    obs, _ = env.reset()
    episode_reward = 0

    for step in range(200):

        action = agent.act(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)

        reward = agent.reward_function(obs, action, next_obs, terminated, truncated)

        episode_reward += reward
        obs = next_obs

        if terminated or truncated:
            break

    scores.append(episode_reward)

    print("Episode", episode+1, "Score:", episode_reward)

mean_score = np.mean(scores)

print("\nAverage Score:", mean_score)

with open("leaderboard/leaderboard.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["MySmartAgent", mean_score])

env.close()