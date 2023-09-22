from __future__ import annotations
from typing import Any

import cvxpy as cp
import numpy as np


class MPCAgent:
    """
    Args:
        environment: An object representing the environment for the agent.
        gamma: A list of discount factors for the objective function.
        safety_margin: A safety margin factor for constraints.
        planning_steps: Number of steps over which to plan.
    """
    def __init__(
        self,
        environment: Any,
        gamma: list[float],
        safety_margin: float = 0.9,
        planning_steps: int = 1,
    ):
        self.gamma: list[float] = gamma
        self.safety_margin: float = safety_margin
        self.planning_steps: int = planning_steps
        self.action_space: Any = environment.action_space
        self.Qlow: np.ndarray = environment.Qlow
        self.num_of_state: int = environment.roomnum
        self.num_of_action: int = environment.action_space.shape[0]
        self.A_d: np.ndarray = environment.A_d
        self.B_d: np.ndarray = environment.B_d
        self.temp: float = environment.OutTemp[environment.epochs]
        self.acmap: np.ndarray = environment.acmap
        self.GroundTemp: float = environment.GroundTemp[environment.epochs]
        self.Occupancy: float = environment.Occupancy[environment.epochs]
        self.ghi: float = environment.ghi[environment.epochs]
        self.target: np.ndarray = environment.target
        self.spacetype: str = environment.spacetype
        self.problem: cp.Problem | None = None

    def predict(self, environment: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            environment: An object representing the environment for the agent.

        Returns:
            tuple: (optimal action, predicted state)
        """
        self.A_d = environment.A_d
        self.B_d = environment.B_d
        self.temp = environment.OutTemp[environment.epochs]
        self.GroundTemp = environment.GroundTemp[environment.epochs]

        self.Occupancy = environment.Occupancy[environment.epochs]
        self.ghi = environment.ghi[environment.epochs]

        action = np.zeros((self.num_of_action))

        x0 = cp.Parameter(self.num_of_state, name="x0")
        u_max = cp.Parameter(self.num_of_action, name="u_max")
        u_min = cp.Parameter(self.num_of_action, name="u_min")

        x = cp.Variable((self.num_of_state, self.planning_steps + 1), name="x")
        u = cp.Variable((self.num_of_action, self.planning_steps), name="u")

        x0.value = environment.state[: self.num_of_state]

        u_max.value = 1.0 * np.ones((self.num_of_action,))
        u_min.value = -1.0 * np.ones((self.num_of_action,))

        x_desired = self.target

        obj = 0
        constr = [x[:, 0] == x0]

        avg_temp = np.sum(x0.value) / self.num_of_state
        Meta = self.Occupancy

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
                == self.A_d @ x[:, t].T
                + self.B_d[:, 3:-1] @ u[:, t]
                + self.B_d[:, 2] * self.temp
                + self.B_d[:, 1] * self.GroundTemp
                + self.B_d[:, 0] * self.Occupower
                + self.B_d[:, -1] * self.ghi,
                u[:, t] <= u_max,
                u[:, t] >= u_min,
            ]

            obj += self.gamma[1] * cp.norm(
                cp.multiply(x[:, t], self.acmap) - x_desired * self.acmap, 2
            ) + self.gamma[0] * 24 * cp.norm(u[:, t], 2)

        prob = cp.Problem(cp.Minimize(obj), constr)

        prob.solve(solver="ECOS_BB")

        state = x[:, 1].value
        if self.spacetype == "continuous":
            action = u.value[:, 0]
        else:
            action = (u.value[:, 0] * 100 - self.Qlow * 100).astype(int)

        return action, state


class MPCAgent_DataDriven:
    """
    Args:
        environment: An object representing the environment for the agent.
        gamma: A list of discount factors for the objective function.
        safety_margin: A safety margin factor for constraints.
        planning_steps: Number of steps over which to plan.
    """
    def __init__(
        self,
        environment: Any,
        gamma: list[float],
        safety_margin: float = 0.9,
        planning_steps: int = 1,
    ):
        self.gamma: list[float] = gamma  # Updated from float to List[float]
        self.Qlow: np.ndarray = environment.Qlow
        self.safety_margin: float = safety_margin
        self.planning_steps: int = planning_steps
        self.action_space: Any = environment.action_space
        self.num_of_state: int = environment.roomnum
        self.num_of_action: int = environment.action_space.shape[0]
        self.A_d: np.ndarray = environment.A_d
        self.B_d: np.ndarray = environment.B_d
        self.temp: float = environment.OutTemp[environment.epochs]
        self.acmap: np.ndarray = environment.acmap
        self.GroundTemp: float = environment.GroundTemp[environment.epochs]
        self.Occupancy: float = environment.Occupancy[environment.epochs]
        self.ghi: float = environment.ghi[environment.epochs]
        self.target: np.ndarray = environment.target
        self.spacetype: str = environment.spacetype
        self.problem: cp.Problem | None = None

    def predict(self, environment: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            environment: An object representing the environment for the agent.

        Returns:
            tuple: (optimal action, predicted state)
        """
        self.A_d = environment.A_d
        self.B_d = environment.B_d
        self.temp = environment.OutTemp[environment.epochs]
        self.GroundTemp = environment.GroundTemp[environment.epochs]

        self.Occupancy = environment.Occupancy[environment.epochs]
        self.ghi = environment.ghi[environment.epochs]

        action = np.zeros((self.num_of_action))

        x0 = cp.Parameter(self.num_of_state, name="x0")
        u_max = cp.Parameter(self.num_of_action, name="u_max")
        u_min = cp.Parameter(self.num_of_action, name="u_max")

        x = cp.Variable((self.num_of_state, self.planning_steps + 1), name="x")
        u = cp.Variable((self.num_of_action, self.planning_steps), name="u")

        x0.value = environment.state[: self.num_of_state]

        u_max.value = 1.0 * np.ones((self.num_of_action,))
        u_min.value = -1.0 * np.ones((self.num_of_action,))

        x_desired = self.target

        obj = 0
        constr = [x[:, 0] == x0]

        avg_temp = np.sum(x0.value) / self.num_of_state
        Meta = self.Occupancy

        for t in range(self.planning_steps):
            constr += [
                x[:, t + 1]
                == self.A_d @ x[:, t].T
                + self.B_d[:, 6:-1] @ u[:, t]
                + self.B_d[:, 5] * self.temp
                + self.B_d[:, 4] * self.GroundTemp
                + self.B_d[:, 3] * Meta
                + self.B_d[:, 2] * Meta**2
                + self.B_d[:, 1] * avg_temp
                + self.B_d[:, 0] * avg_temp**2
                + self.B_d[:, -1] * self.ghi,
                u[:, t] <= u_max,
                u[:, t] >= u_min,
            ]

            obj += self.gamma[1] * cp.norm(
                cp.multiply(x[:, t], self.acmap) - x_desired * self.acmap, 2
            ) + self.gamma[0] * 24 * cp.norm(u[:, t], 2)

        prob = cp.Problem(cp.Minimize(obj), constr)

        prob.solve(solver="ECOS_BB")

        state = x[:, 1].value
        if self.spacetype == "continuous":
            action = u.value[:, 0]
        else:
            action = (u.value[:, 0] * 100 - self.Qlow * 100).astype(int)

        return action, state
