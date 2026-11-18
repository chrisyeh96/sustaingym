from gymnasium.envs.registration import register

register(
    id='sustaingym/EVCharging-v0',
    entry_point='sustaingym.envs.evcharging:EVChargingEnv',
    nondeterministic=False
)

# register(
#     id='sustaingym/ElectricityMarket-v0',
#     entry_point='sustaingym.envs.electricitymarket:ElectricityMarketEnv'
# )

register(
    id='sustaingym/DataCenter-v0',
    entry_point='sustaingym.envs.data_center:DataCenterEnv',
    nondeterministic=False
)

register(
    id='sustaingym/Building-v0',
    entry_point='sustaingym.envs.building:BuildingEnv',
    nondeterministic=False
)

register(
    id='sustaingym/Cogen-v0',
    entry_point='sustaingym.envs.cogen:CogenEnv',
    nondeterministic=False
)
