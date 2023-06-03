# TO RUN, copy-paste in terminal:
# ./sustaingym/scripts/evcharging/run_train_rllib_lr5e-5.sh


# --- PPO ---

# # DONE: Single-agent continuous
# python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 123 --lr 5e-5

# # DONE: Single-agent discrete
# python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 123 --lr 5e-5 -d

# NOT WORKING: Multi-agent continuous
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 123 --lr 5e-5 -m

# NOT WORKING: Multi-agent discrete
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 123 --lr 5e-5 -m -d


# --- SAC ---

# # DONE: Single-agent continuous
# python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 123 --lr 5e-5

# NOT WORKING: Single-agent discrete
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 123 --lr 5e-5 -d

# NOT WORKING: Multi-agent continuous
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 123 --lr 5e-5 -m

# NOT WORKING: Multi-agent discrete
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 123 --lr 5e-5 -m -d
