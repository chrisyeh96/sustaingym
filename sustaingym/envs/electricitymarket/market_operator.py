"""Implements MarketOperator class, which solves the security-constrained
economic dispatch (SCED) problem.

Abbreviations:
- MILP = "mixed-integer linear program"
- SOC = "state of charge" of battery (MWh)
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from .networks import Network
from ..utils import solve_mosek


class MarketOperator:
    """MarketOperator class."""
    def __init__(self, network: Network, congestion: bool,
                 time_step_duration: float, h: int,
                 milp: bool = True):
        """
        Args:
            network: Network, defines network properties
            congestion: bool, whether to include congestion constraints
            time_step_duration: float, length of a time step in hours
            h: int, horizon (settlement interval) in # of time steps, >= 0
            milp: bool, whether to solve a mixed-integer linear program to
                enforce that charge * discharge = 0
        """
        self.network = network
        self.congestion = congestion
        self.time_step_duration = time_step_duration
        self.h = h
        self.milp = milp

        net = network
        N_D = net.N_D  # num loads
        N_G = net.N_G  # num generators
        N_B = net.N_B  # num batteries

        # Variables
        self.soc = cp.Variable((N_B, h + 2), name='soc')  # state of charge (MWh)
        self.dispatch = cp.Variable((N_D + N_G + 2*N_B, h + 1), name='dispatch')  # dispatch (MW)
        demand_dis = self.dispatch[:N_D]  # demands (loads, MW)
        gen_dis = self.dispatch[N_D: N_D+N_G]  # generator dispatch (MW)
        bat_c = self.dispatch[N_D+N_G: N_D+N_G+N_B]  # battery charge (MW)
        bat_d = self.dispatch[-N_B:]  # battery discharge (MW)

        # Parameters
        # bat_c_max and bat_d_max are constants provided by the network, but
        # we need to make them parameters with a time dimension so that they
        # can be modified by the MILP to enforce that the battery is either
        # charging or discharging, not both.
        self.demand = cp.Parameter((N_D, h + 1), name='demand forecast')  # (MW)
        self.bat_c_max = cp.Parameter((N_B, h + 1), name='battery max charge')  # (MW)
        self.bat_d_max = cp.Parameter((N_B, h + 1), name='battery max discharge')  # (MW)
        self.c_bat_c = cp.Parameter((N_B, h + 1), name='battery cost charge')  # charge cost ($/MWh)
        self.c_bat_d = cp.Parameter((N_B, h + 1), name='battery cost discharge')  # discharge cost ($/MWh)
        self.soc_init = cp.Parameter(N_B, nonneg=True, name='soc initial')  # initial SOC (MWh)
        self.soc_final = cp.Parameter(N_B, nonneg=True, name='soc final')  # final SOC (MWh)

        # Constraints
        constraints = [
            # demand is fixed
            demand_dis == self.demand,

            # generation limits
            gen_dis >= net.g_min[:, None],
            gen_dis <= net.g_max[:, None],

            # battery limits
            bat_c >= 0,
            bat_c <= self.bat_c_max,
            bat_d >= 0,
            bat_d <= self.bat_d_max,

            # battery range
            self.soc >= net.soc_min,
            self.soc <= net.soc_max,

            # initial soc
            self.soc[:, 0] == self.soc_init,

            # charging dynamics
            self.soc[:, 1:] == (
                self.soc[:, :-1]
                + cp.multiply(net.eff_c, bat_c) * time_step_duration
                - cp.multiply(1. / net.eff_d, bat_d) * time_step_duration),
        ]

        # power balance
        self.power_balance_constr: cp.Constraint = (
            cp.sum(net.J @ self.dispatch * time_step_duration, axis=0) == 0
        )
        constraints.append(self.power_balance_constr)

        if self.congestion:
            # power flow and line flow limits (aka congestion constraints)
            Hp = net.H @ net.J @ self.dispatch  # (in MW)
            self.congestion_constrs = [
                Hp <= net.f_max[:, None],
                Hp >= -net.f_max[:, None],
            ]
            constraints.extend(self.congestion_constrs)

        # Objective function
        obj = 0
        for tau in range(h + 1):
            # add up all generator costs
            # - net.c_gen[:, 0] is a constant, so we can ignore it in the objective
            obj += (
                net.c_gen[:, 1] @ gen_dis[:, tau]        # generators
                + self.c_bat_d[:, tau] @ bat_d[:, tau]   # battery discarge
                - self.c_bat_c[:, tau] @ bat_c[:, tau])  # battery charge

        self.obj = cp.Minimize(obj)
        self.prob = cp.Problem(self.obj, constraints)
        assert self.prob.is_dcp() and self.prob.is_dpp()

        if milp:
            # use mixed-integer linear program (MILP) to enforce that
            # battery is either only charging or only discharging (not both)
            # at every time step
            self.z = cp.Variable((N_B, h + 1), boolean=True)  # z ∈ {0,1}
            milp_constraints = [
                bat_c <= cp.multiply(self.bat_c_max, self.z),
                bat_d <= cp.multiply(self.bat_d_max, 1 - self.z),
            ]
            self.milp_prob = cp.Problem(
                self.obj, constraints=constraints + milp_constraints)
            assert self.milp_prob.is_dcp() and self.milp_prob.is_dpp()

    def get_dispatch(self, demand: np.ndarray, demand_forecast: np.ndarray,
                     c_bat: np.ndarray | None, soc: np.ndarray | None,
                     soc_final: np.ndarray | None, verbose: bool = False
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Determines dispatch values.

        Args:
            demand: array, shape [N_D], nodal demands (MW)
            demand_forecast: array, shape [N_D, h], nodal demands forecast (MW)
            c_bat: array or None, shape [2, N_B, h + 1],
                c_bat[0] is battery charge bid price ($/MWh),
                c_bat[1] is battery discharge cost ($/MWh),
                set to None to have no battery participation
            soc: array or None, shape [N_B], current SOC of batteries (MWh)
                set to None to have no battery participation
            soc_final: array or None, shape [N_B], desired SOC at end of SCED
                horizon
                set to None to have no battery participation
            verbose: bool, whether to print detailed outputs

        Returns:
            gen_dis: array, shape [N_G], generator dispatch (MW)
            bat_d: array, shape [N_B], battery discharge amounts (MWh)
            bat_c: array, shape [N_B], battery charge amounts (MWh)
            prices: array, shape [N], nodal prices ($/MWh)
        """
        net = self.network
        N = net.N      # num buses
        N_D = net.N_D  # num loads
        N_G = net.N_G  # num generators
        N_B = net.N_B  # num batteries
        h = self.h     # horizon

        # shape [N_D, h+1], units MW
        self.demand.value = np.concatenate(
            [demand[:, None], demand_forecast], axis=1)

        if c_bat is None:
            assert soc is None
            assert soc_final is None
            self.bat_c_max.value = np.zeros([N_B, h + 1])
            self.bat_d_max.value = np.zeros([N_B, h + 1])
            self.soc_init.value = np.zeros(N_B)
            self.soc_final.value = np.zeros(N_B)
            self.c_bat_c.value = np.zeros([N_B, h + 1])
            self.c_bat_d.value = np.zeros([N_B, h + 1])
        else:
            assert soc is not None
            assert soc_final is not None
            self.bat_c_max.value = np.tile(net.bat_c_max, (1, h+1))
            self.bat_d_max.value = np.tile(net.bat_d_max, (1, h+1))
            self.soc_init.value = soc
            self.soc_final.value = soc_final
            # c_bat has shape [2, N_B, h + 1]
            self.c_bat_c.value = c_bat[0, :, :]
            self.c_bat_d.value = c_bat[1, :, :]

        # solve optimization problem
        if self.milp:
            # if using MILP, solve it first. However, the MILP does not use
            # dual variables like a regular linear program. So we use the
            # result from the MILP to constrain the battery charge/discharge
            # upper-bound. Finally, resolve the linear program to get the
            # appropriate dual variables. This resolve will be very fast,
            # because the variables will already be initialized to the optimal
            # values from the MILP.
            solve_mosek(self.milp_prob)
            self.bat_c_max.value *= self.z.value
            self.bat_d_max.value *= 1 - self.z.value

            # ensure power maxes for battery are within the expected bounds
            self.bat_c_max.value = self.bat_c_max.value.clip(
                min=0., max=net.bat_c_max)
            self.bat_d_max.value = self.bat_d_max.value.clip(
                min=0., max=net.bat_d_max)

        solve_mosek(self.prob)

        # debugging
        msg: str | None = None
        if self.prob.status == 'infeasible':
            msg = 'prob infeasible'
        if self.power_balance_constr.dual_value is None:
            msg = 'missing power balance constraint dual variable'
        if msg is not None:
            all_vars = locals()
            import pickle
            with open('fail.pkl', 'wb') as f:
                pickle.dump(all_vars, f)
            import pdb
            pdb.set_trace()

        assert self.power_balance_constr.dual_value is not None
        lam = -self.power_balance_constr.dual_value[0]

        if self.congestion:
            mu_plus = self.congestion_constrs[0].dual_value[:, 0]
            mu_minus = self.congestion_constrs[1].dual_value[:, 0]
            prices = lam + net.H.T @ (mu_plus - mu_minus)
        else:
            prices = lam * np.ones(N)

        dispatch = self.dispatch.value
        gen_dis = dispatch[N_D:N_D+N_G]  # generator dispatch (MW)
        bat_c = dispatch[N_D+N_G: N_D+N_G+N_B] * self.time_step_duration  # charge (MWh)
        bat_d = dispatch[-N_B:] * self.time_step_duration  # discharge (MWh)

        if verbose:
            Δsoc = self.soc.value[:, 1:] - self.soc[:, :-1]  # shape [N_B, h+1]

            with np.printoptions(precision=2):
                print(f'========== dispatch output ==========')
                print(f'network: {N} buses, {net.M} lines, {N_D} loads, '
                      f'{N_G} generators, {N_B} batteries')
                print(f'SCED horizon: {h}')
                print(f'total load (MW): {self.demand.value.sum(axis=0)}')
                print(f'objective value ($): {self.prob.value}')
                print(f'battery soc initial (MWh): {self.soc_init.value}, final target: ≥{self.soc_final.value}')
                print(f'battery soc (MWh): {self.soc.value}')
                print(f'battery Δsoc (MWh): {Δsoc}')
                print(f'battery discharge (MWh): {bat_d}')
                print(f'battery charge (MWh): {bat_c}')
                print(f'battery charge costs ($/Mwh): {self.c_bat_c.value}')
                print(f'battery discharge costs ($/Mwh): {self.c_bat_d.value}')
                print(f'generator dispatch (MW): {gen_dis}')
                print(f'generator costs ($/MWh): {net.c_gen[:, 1]}')

                congested = False
                if self.congestion:
                    congested_days = (
                        (self.congestion_constrs[0].dual_value > 1e-5).any(axis=0)
                        | (self.congestion_constrs[1].dual_value > 1e-5).any(axis=0))
                    print(f'congested: {congested_days}')
                    congested = congested_days.any()
                if congested:
                    print(f'nodal prices ($/MWh): {prices}')
                else:
                    print(f'uniform price ($/MWh): {lam}')

                print()

        return gen_dis[:, 0], bat_d[:, 0], bat_c[:, 0], prices
