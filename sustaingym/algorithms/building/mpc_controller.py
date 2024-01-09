from __future__ import annotations

import cvxpy as cp
import numpy as np

from sustaingym.envs.building import BuildingEnv


class MPCAgent:
    """
    Args:
        env: An object representing the environment for the agent.
        beta: temperature error penalty weight for reward function
        pnorm: p to use for norm in reward function
        safety_margin: A safety margin factor for constraints.
        planning_steps: Number of steps over which to plan.

    TODO: rewrite using cp.Parameters
    """
    def __init__(
        self,
        env: BuildingEnv,
        beta: float,
        pnorm: float,
        safety_margin: float = 0.9,
        planning_steps: int = 1,
    ):
        self.env = env
        self.beta = beta
        self.pnorm = pnorm
        self.safety_margin = safety_margin
        self.planning_steps = planning_steps

        self.action_dim: int = env.action_space.shape[0]
        self.temp: float = env.out_temp[env.epoch]
        self.ground_temp: float = env.ground_temp[env.epoch]
        self.metabolism: float = env.metabolism[env.epoch]
        self.ghi: float = env.ghi[env.epoch]

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            env: An object representing the environment for the agent.

        Returns:
            tuple: (optimal action, predicted state)
        """
        env = self.env
        A_d = env.A_d
        B_d = env.B_d
        n = env.n

        self.temp = env.out_temp[env.epoch]
        self.ground_temp = env.ground_temp[env.epoch]

        self.metabolism = env.metabolism[env.epoch]
        self.ghi = env.ghi[env.epoch]

        action = np.zeros(self.action_dim)

        x0 = cp.Parameter(n, name="x0")
        u_max = cp.Parameter(self.action_dim, name="u_max")
        u_min = cp.Parameter(self.action_dim, name="u_min")

        x = cp.Variable((n, self.planning_steps + 1), name="x")
        u = cp.Variable((self.action_dim, self.planning_steps), name="u")

        x0.value = env.state[:n]

        u_max.value = 1.0 * np.ones(self.action_dim)
        u_min.value = -1.0 * np.ones(self.action_dim)

        x_desired = env.target

        obj = 0
        constr = [x[:, 0] == x0]

        avg_temp = np.sum(x0.value) / n
        Meta = self.metabolism

        self.Occupower = (
            6.461927
            + 0.946892 * Meta
            + 0.0000255737 * Meta**2
            - 0.0627909 * avg_temp * Meta
            + 0.0000589172 * avg_temp * Meta**2
            - 0.19855 * avg_temp**2
            + 0.000940018 * avg_temp**2 * Meta
            - 0.00000149532 * avg_temp**2 * Meta**2
        )

        for t in range(self.planning_steps):
            constr += [
                x[:, t + 1]
                == A_d @ x[:, t].T
                + B_d[:, 3:-1] @ u[:, t]
                + B_d[:, 2] * self.temp
                + B_d[:, 1] * self.ground_temp
                + B_d[:, 0] * self.Occupower
                + B_d[:, -1] * self.ghi,
                u[:, t] <= u_max,
                u[:, t] >= u_min,
            ]

            obj += self.beta * cp.norm(
                cp.multiply(x[:, t], env.ac_map) - x_desired * env.ac_map, self.pnorm
            ) + (1 - self.beta) * 24 * cp.norm(u[:, t], self.pnorm)

        prob = cp.Problem(cp.Minimize(obj), constr)

        prob.solve(solver="ECOS_BB")

        state = x[:, 1].value
        if env.is_continuous_action:
            action = u.value[:, 0]
        else:
            action = (u.value[:, 0] * 100 - env.Qlow * 100).astype(int)

        return action, state


class MPCAgent_DataDriven:
    """
    Args:
        env: An object representing the environment for the agent.
        beta: temperature error penalty weight for reward function
        pnorm: p to use for norm in reward function
        safety_margin: A safety margin factor for constraints.
        planning_steps: Number of steps over which to plan.

    TODO: rewrite using cp.Parameters
    """
    def __init__(
        self,
        env: BuildingEnv,
        beta: float,
        pnorm: float,
        safety_margin: float = 0.9,
        planning_steps: int = 1,
    ):
        self.env = env
        self.beta = beta
        self.pnorm = pnorm
        self.safety_margin = safety_margin
        self.planning_steps = planning_steps

        self.action_dim: int = env.action_space.shape[0]
        self.temp: float = env.out_temp[env.epoch]
        self.ground_temp: float = env.ground_temp[env.epoch]
        self.metabolism: float = env.metabolism[env.epoch]
        self.ghi: float = env.ghi[env.epoch]

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            env: An object representing the environment for the agent.

        Returns:
            tuple: (optimal action, predicted state)
        """
        env = self.env
        A_d = env.A_d
        B_d = env.B_d
        n = env.n

        self.temp = env.out_temp[env.epoch]
        self.ground_temp = env.ground_temp[env.epoch]

        self.metabolism = env.metabolism[env.epoch]
        self.ghi = env.ghi[env.epoch]

        action = np.zeros(self.action_dim)

        x0 = cp.Parameter(n, name="x0")
        u_max = cp.Parameter(self.action_dim, name="u_max")
        u_min = cp.Parameter(self.action_dim, name="u_max")

        x = cp.Variable((n, self.planning_steps + 1), name="x")
        u = cp.Variable((self.action_dim, self.planning_steps), name="u")

        x0.value = env.state[:n]

        u_max.value = 1.0 * np.ones(self.action_dim)
        u_min.value = -1.0 * np.ones(self.action_dim)

        x_desired = env.target

        obj = 0
        constr = [x[:, 0] == x0]

        avg_temp = np.sum(x0.value) / n
        Meta = self.metabolism

        for t in range(self.planning_steps):
            constr += [
                x[:, t + 1]
                == A_d @ x[:, t].T
                + B_d[:, 6:-1] @ u[:, t]
                + B_d[:, 5] * self.temp
                + B_d[:, 4] * self.ground_temp
                + B_d[:, 3] * Meta
                + B_d[:, 2] * Meta**2
                + B_d[:, 1] * avg_temp
                + B_d[:, 0] * avg_temp**2
                + B_d[:, -1] * self.ghi,
                u[:, t] <= u_max,
                u[:, t] >= u_min,
            ]

            obj += self.beta * cp.norm(
                cp.multiply(x[:, t], env.ac_map) - x_desired * env.ac_map, self.pnorm
            ) + (1 - self.beta) * 24 * cp.norm(u[:, t], self.pnorm)

        prob = cp.Problem(cp.Minimize(obj), constr)

        prob.solve(solver="ECOS_BB")

        state = x[:, 1].value
        if env.is_continuous_action:
            action = u.value[:, 0]
        else:
            action = (u.value[:, 0] * 100 - env.Qlow * 100).astype(int)

        return action, state
