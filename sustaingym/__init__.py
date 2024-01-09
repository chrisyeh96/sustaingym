from gymnasium.envs.registration import register

register(
    id='sustaingym/EVCharging-v0',
    entry_point='sustaingym.envs:EVChargingEnv'
)

register(
    id='sustaingym/ElectricityMarket-v0',
    entry_point='sustaingym.envs:ElectricityMarketEnv'
)

register(
    id='sustaingym/DataCenter-v0',
    entry_point='sustaingym.envs:DataCenterEnv'
)
