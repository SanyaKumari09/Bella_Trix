import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MysteryControlEnv(gym.Env):

    def __init__(self):

        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0, high=100, shape=(4,), dtype=np.float32
        )

        self.max_steps = 200
        self.current_step = 0

        self.inlet_flow_rate = 10
        self.outlet_flow_rate = 8
        self.heat_coefficient = 5
        self.cooling_coefficient = 2

        self.state = None

    def reset(self, seed=None, options=None):

        pressure = np.random.uniform(20, 40)
        temp = np.random.uniform(20, 30)

        target_pressure = np.random.uniform(50, 70)
        target_temp = np.random.uniform(60, 80)

        self.state = np.array(
            [pressure, temp, target_pressure, target_temp],
            dtype=np.float32,
        )

        self.current_step = 0

        return self.state, {}

    def step(self, action):

        inlet_v, outlet_v, heater_p = action

        pressure, temp, target_pressure, target_temp = self.state

        pressure_change = (inlet_v * self.inlet_flow_rate) - (outlet_v * self.outlet_flow_rate)

        new_pressure = np.clip(pressure + pressure_change, 0, 100)

        temp_change = (heater_p * self.heat_coefficient) - (self.cooling_coefficient * (temp / 100))

        new_temp = np.clip(temp + temp_change, 0, 100)

        terminated = False

        if new_pressure > 95 or new_temp > 95:
            terminated = True

        pressure_error = abs(new_pressure - target_pressure)
        temp_error = abs(new_temp - target_temp)

        reward = -(pressure_error + temp_error)

        if terminated:
            reward -= 100

        self.state = np.array(
            [new_pressure, new_temp, target_pressure, target_temp],
            dtype=np.float32,
        )

        self.current_step += 1

        truncated = self.current_step >= self.max_steps

        return self.state, reward, terminated, truncated, {}