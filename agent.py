from agent_template import ParticipantAgent
import numpy as np


class MySmartAgent(ParticipantAgent):

    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)

        # PID memory
        self.prev_p_error = 0
        self.prev_t_error = 0

        self.integral_p = 0
        self.integral_t = 0

        # smoothing memory
        self.prev_action = np.array([0.0, 0.0, 0.0])

    def act(self, observation):

        pressure, temp, target_pressure, target_temp = observation

        p_error = target_pressure - pressure
        t_error = target_temp - temp

        # adaptive PID gains
        if abs(p_error) > 20:
            kp_p = 0.08
        else:
            kp_p = 0.05

        if abs(t_error) > 20:
            kp_t = 0.08
        else:
            kp_t = 0.05

        ki = 0.0005
        kd = 0.02

        # integral accumulation
        self.integral_p += p_error
        self.integral_t += t_error

        # anti-windup
        self.integral_p = np.clip(self.integral_p, -100, 100)
        self.integral_t = np.clip(self.integral_t, -100, 100)

        # derivative
        d_p = p_error - self.prev_p_error
        d_t = t_error - self.prev_t_error

        pressure_control = kp_p*p_error + ki*self.integral_p + kd*d_p
        temp_control = kp_t*t_error + ki*self.integral_t + kd*d_t

        inlet = np.clip(pressure_control, 0, 1)
        outlet = np.clip(-pressure_control, 0, 1)
        heater = np.clip(temp_control, 0, 1)

        # safety controller
        if pressure > 90:
            inlet = 0
            outlet = 1

        if temp > 90:
            heater = 0

        action = np.array([inlet, outlet, heater])

        # smooth actions
        max_change = 0.2
        action = np.clip(
            action,
            self.prev_action - max_change,
            self.prev_action + max_change
        )

        action = np.clip(action, 0, 1)

        self.prev_action = action

        self.prev_p_error = p_error
        self.prev_t_error = t_error

        return action.astype(np.float32)

    def reward_function(self, state, action, next_state, terminated, truncated):

        pressure, temp, target_pressure, target_temp = state
        next_pressure, next_temp, _, _ = next_state

        p_error = abs(next_pressure - target_pressure)
        t_error = abs(next_temp - target_temp)

        reward = -(p_error * 0.6 + t_error * 0.4)

        # stability bonus
        if p_error < 2:
            reward += 10

        if t_error < 2:
            reward += 10

        if p_error < 1 and t_error < 1:
            reward += 20

        # energy penalty
        reward -= 0.05 * np.sum(np.abs(action))

        if terminated:
            reward -= 500

        return reward