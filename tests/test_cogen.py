from __future__ import annotations

import unittest

import gymnasium.utils.env_checker
from pettingzoo.test import parallel_api_test
# from pettingzoo.test.seed_test import parallel_seed_test
import ray.rllib.utils

from sustaingym.envs.cogen import (
    CogenEnv, MultiAgentCogenEnv, MultiAgentRLLibCogenEnv)


class TestSingleAgentEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = CogenEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_check_env(self):
        gymnasium.utils.env_checker.check_env(self.env)

    def test_rllib_check_env(self) -> None:
        ray.rllib.utils.check_env(self.env)


class TestMultiAgentEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MultiAgentCogenEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_pettingzoo_parallel_api(self) -> None:
        parallel_api_test(self.env, num_cycles=1000)

    # this test fails in PettingZoo v1.22.3 due to a PettingZoo bug
    # https://github.com/Farama-Foundation/PettingZoo/issues/939
    # TODO: uncomment once we upgrade to PettingZoo >= 1.23
    # def test_pettingzoo_parallel_seed(self) -> None:
    #     parallel_seed_test(MultiAgentCogenEnv)


class TestMultiAgentRLLibEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MultiAgentRLLibCogenEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_rllib_check_env(self) -> None:
        ray.rllib.utils.check_env(self.env)


if __name__ == '__main__':
    unittest.main()
