# ./sustaingym/scripts/evcharging/run_train_rllib_2019_1.sh


# --- PPO ---


# Single-agent continuous
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2019 -s caltech -r 123 --lr ???

# Single-agent discrete
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2019 -s caltech -r 123 --lr ??? -d

# Multi-agent continuous
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2019 -s caltech -r 123 --lr ??? -m

# Multi-agent discrete
python -m sustaingym.scripts.evcharging.train_rllib -a ppo -t Summer 2019 -s caltech -r 123 --lr ??? -m -d


# --- SAC ---

# Single-agent continuous
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2019 -s caltech -r 123 --lr ???

# Single-agent discrete
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2019 -s caltech -r 123 --lr ??? -d

# Multi-agent continuous
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2019 -s caltech -r 123 --lr ??? -m

# Multi-agent discrete
python -m sustaingym.scripts.evcharging.train_rllib -a sac -t Summer 2019 -s caltech -r 123 --lr ??? -m -d
