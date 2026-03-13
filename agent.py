from agent_template import ParticipantAgent
import numpy as np

class MySmartAgent(ParticipantAgent):

    def act(self, observation):

        pressure, temp, target_pressure, target_temp = observation

        p_diff = target_pressure - pressure
        t_diff = target_temp - temp

        inlet = np.clip(p_diff / 30, 0, 1)
        outlet = np.clip(-p_diff / 30, 0, 1)

        heater = np.clip(t_diff / 30, 0, 1)

        return np.array([inlet, outlet, heater], dtype=np.float32)

    def reward_function(self, state, action, next_state, terminated, truncated):

        pressure, temp, target_pressure, target_temp = state
        next_pressure, next_temp, _, _ = next_state

        p_error = abs(next_pressure - target_pressure)
        t_error = abs(next_temp - target_temp)

        reward = -(p_error * 0.7 + t_error * 0.3)

        if p_error < 2:
            reward += 10

        if t_error < 2:
            reward += 10

        if terminated:
            reward -= 500

        return reward