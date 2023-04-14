a = 

# python -m sustaingym.scripts.evcvharging.run_train_stable_baselines
    
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description='train RLLib models on EVChargingEnv',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument(
#         '-a', '--algo', type=str, default='ppo',
#         help="'ppo', 'a2c', or 'sac'")
#     parser.add_argument(
#         '-t', '--train_date_period', type=str, default='Summer 2019',
#         help="'Summer 2019' or 'Summer 2021'")
#     parser.add_argument(
#         '-s', '--site', type=str, default='caltech',
#         help='site of garage. caltech or jpl')
#     parser.add_argument('-d', '--discrete', action=argparse.BooleanOptionalAction)
#     parser.add_argument('-m', '--multiagent', action=argparse.BooleanOptionalAction)
#     parser.add_argument(
#         '-p', '--periods_delay', type=int, default=0,
#         help='communication delay in multiagent setting. Ignored for single agent.')
#     parser.add_argument(
#         '-s', '--seed', type=int, default=0,
#         help='Random seed')
#     args = parser.parse_args()

#     config = {
#         "algo": args.algo,
#         "dp": args.train_date_period,
#         "site": args.site,
#         "discrete": args.discrete,
#         "multiagent": args.multiagent,
#         "periods_delay": args.periods_delay,
#         "seed": args.seed
#     }