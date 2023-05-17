from sustaingym.envs.evcharging import GMMsTraceGenerator, RealTraceGenerator
from sustaingym.envs.evcharging import EVChargingEnv

import collections
import numpy as np
import random
random.seed(42)


# Test trace generation
def validate_trace_generation() -> None:
    in_covid = ('2020-02-01', '2020-05-31')
    pre_covid = ('2019-05-01', '2019-08-31')
    pre_covid_str = 'Summer 2019'

    atg1 = GMMsTraceGenerator('caltech', date_period='Summer 2019', n_components=30, period=5)
    atg2 = GMMsTraceGenerator('caltech', date_period='Summer 2019', n_components=30, period=10)
    rtg3 = RealTraceGenerator('jpl', date_period='Spring 2020', period=5)
    rtg4 = RealTraceGenerator('jpl', date_period='Spring 2020', period=10)

    avg_events = [0] * 4
    generators = [atg1, atg2, rtg3, rtg4]
    for seed in range(5):
        for i in range(4):
            generator = generators[i]
            generator.set_random_seed(seed)
            eq, evs, num_events = generator.get_event_queue()
            a = generator.get_moer()
            b, c = a[240, 1], a[20, 0]
            print("num events: ", num_events, b, c, evs[0].arrival, evs[1].departure)
            avg_events[i] += num_events
    print('num events avg: ', [avg_event / 5 for avg_event in avg_events])


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
    print("Validating event generation: ")
    validate_trace_generation()
    print("Validating environment: ")
    validate_environment()
