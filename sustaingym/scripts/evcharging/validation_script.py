from sustaingym.envs.evcharging import GMMsTraceGenerator, RealTraceGenerator
from sustaingym.envs.evcharging import EVChargingEnv

import collections
import numpy as np
import random
random.seed(42)


def get_action(num_stations: int, type: str = 'random') -> np.ndarray:
    if type == 'random':
        return np.random.randint(size=num_stations, low=0, high=5)
    elif type == 'full':
        return np.full(num_stations, 4)
    elif type == 'empty':
        return np.full(num_stations, 0)
    elif type == 'half':
        return np.full(num_stations, 2)
    else:
        raise ValueError('Bad action')


def validate_environment() -> None:
    for policy in ['random', 'full', 'empty', 'half']:
        print(f'--- policy: {policy} ---')
        generator = RealTraceGenerator(site='jpl', date_period='Summer 2019', sequential=True, period=5, random_seed=100)
        gen_type = 'real'
        for action_type in ['discrete', 'continuous']:
            for project_action in [True, False]:
                env = EVChargingEnv(generator, action_type, project_action)

                for seed in [42, 43, 44]:
                    observation = env.reset(seed=seed)
                    print('gen_type, action, projection, num evs: ', gen_type, action_type, project_action, len(env.evs))

                    rewards = 0.
                    done = False
                    i = 0
                    d = collections.defaultdict(int)
                    while not done:
                        action = get_action(env.num_stations, policy)
                        observation, reward, done, info = env.step(action)

                        rewards += reward
                        for k in info['reward']:
                            d[k] += info['reward'][k]
                        i += 1
                    print("reward, num timesteps: ", rewards, i)
                    env.close()


if __name__ == '__main__':
    print("Validating environment: ")
    validate_environment()
