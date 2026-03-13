from control_env import MysteryControlEnv
from agent import MySmartAgent

env = MysteryControlEnv()

agent = MySmartAgent(env.action_space, env.observation_space)

episodes = 10
total_score = 0

for i in range(episodes):

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

    print("Episode", i+1, "Score:", episode_reward)

    total_score += episode_reward


average_score = total_score / episodes

print("Average Score:", average_score)

env.close()