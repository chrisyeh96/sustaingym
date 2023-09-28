from __future__ import annotations

import unittest

import gymnasium.utils.env_checker
from pettingzoo.test import parallel_api_test
from pettingzoo.test.seed_test import parallel_seed_test
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import ray.rllib.utils

from sustaingym.envs.building import (
    BuildingEnv, MultiAgentBuildingEnv, ParameterGenerator)


class TestSingleAgentEnv(unittest.TestCase):
    def setUp(self) -> None:
        params = ParameterGenerator(
            building='OfficeSmall', weather='Hot_Dry', location='Tucson')
        self.env = BuildingEnv(params)

    def tearDown(self) -> None:
        self.env.close()

    def test_check_env(self):
        gymnasium.utils.env_checker.check_env(self.env)

    def test_rllib_check_env(self) -> None:
        ray.rllib.utils.check_env(self.env)


class TestMultiAgentEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.params = ParameterGenerator(
            building='OfficeSmall', weather='Hot_Dry', location='Tucson')
        self.env = MultiAgentBuildingEnv(self.params)

    def tearDown(self) -> None:
        self.env.close()

    def test_check_env(self) -> None:
        parallel_api_test(self.env, num_cycles=1000)

    def test_parallel_seed(self) -> None:
        parallel_seed_test(lambda: MultiAgentBuildingEnv(self.params))

    # this test fails in RLLib v2.6.3 due to several RLLib bugs:
    # - https://github.com/ray-project/ray/pull/39431
    # - https://github.com/ray-project/ray/issues/39453
    # this should be fixed in RLLib v2.8
    # def test_rllib_check_env(self) -> None:
    #     rllib_env = ParallelPettingZooEnv(self.env)
    #     ray.rllib.utils.check_env(rllib_env)


if __name__ == '__main__':
    unittest.main()
