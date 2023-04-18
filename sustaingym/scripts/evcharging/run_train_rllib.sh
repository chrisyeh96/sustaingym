# ./sustaingym/scripts/evcharging/run_train_rllib.sh

# # out of dist
# python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2019 -s caltech -r 123
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2019 -s caltech -r 246
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2019 -s caltech -r 369
# python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2019 -s caltech -r 123
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2019 -s caltech -r 246
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2019 -s caltech -r 369
# python -m sustaingym.scripts.evcharging.train_rllib -a a2c -t Summer 2019 -s caltech -r 123
python -m sustaingym.scripts.evcharging.train_rllib -a a2c -t Summer 2019 -s caltech -r 246
python -m sustaingym.scripts.evcharging.train_rllib -a a2c -t Summer 2019 -s caltech -r 369

# # in dist
# python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 123
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 246
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 369
# python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 123
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 246
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 369
# python -m sustaingym.scripts.evcharging.train_rllib -a a2c -t Summer 2021 -s caltech -r 123
python -m sustaingym.scripts.evcharging.train_rllib -a a2c -t Summer 2021 -s caltech -r 246
python -m sustaingym.scripts.evcharging.train_rllib -a a2c -t Summer 2021 -s caltech -r 369


# multiagent
# out of dist
# python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2019 -s caltech -r 123 -m
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2019 -s caltech -r 246 -m
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2019 -s caltech -r 369 -m
# python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2019 -s caltech -r 123 -m
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2019 -s caltech -r 246 -m
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2019 -s caltech -r 369 -m
# python -m sustaingym.scripts.evcharging.train_rllib -a a2c -t Summer 2019 -s caltech -r 123 -m
python -m sustaingym.scripts.evcharging.train_rllib -a a2c -t Summer 2019 -s caltech -r 246 -m
python -m sustaingym.scripts.evcharging.train_rllib -a a2c -t Summer 2019 -s caltech -r 369 -m

# in dist
# python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 123 -m
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 246 -m
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 369 -m
# python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 123 -m
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 246 -m
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 369 -m
# python -m sustaingym.scripts.evcharging.train_rllib -a a2c -t Summer 2021 -s caltech -r 123 -m
python -m sustaingym.scripts.evcharging.train_rllib -a a2c -t Summer 2021 -s caltech -r 246 -m
python -m sustaingym.scripts.evcharging.train_rllib -a a2c -t Summer 2021 -s caltech -r 369 -m
