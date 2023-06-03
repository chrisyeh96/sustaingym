# ./sustaingym/scripts/evcharging/run_train_rllib_lr5e-6.sh


# --- PPO ---

# Single-agent continuous
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 123 --lr 5e-6

# Single-agent discrete
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 123 --lr 5e-6 -d

# Multi-agent continuous
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 123 --lr 5e-6 -m

# Multi-agent discrete
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2021 -s caltech -r 123 --lr 5e-6 -m -d


# --- SAC ---

# Single-agent continuous
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 123 --lr 5e-6

# Single-agent discrete
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 123 --lr 5e-6 -d

# Multi-agent continuous
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 123 --lr 5e-6 -m

# Multi-agent discrete
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2021 -s caltech -r 123 --lr 5e-6 -m -d
