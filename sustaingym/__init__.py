from gym.envs.registration import register

register(
    id='sustaingym/EVCharging-v0',
    entry_point='sustaingym.envs:EVChargingEnv'
)

register(
    id='sustaingym/BatteryStorage-v0',
    entry_point='sustaingym.envs:BatteryStorageEnv'
)

register(
    id='sustaingym/DataCenter-v0',
    entry_point='sustaingym.envs:DataCenterEnv'
)

# register(
#     id='sustaingym/ElectricityMarket-v0',
#     entry_point='sustaingym.envs:ElectricityMarketEnv'
# )

# register(
#     id='sustaingym/CongestedElectricityMarket-v0',
#     entry_point='sustaingym.envs:CongestedElectricityMarketEnv'
# )
