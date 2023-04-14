# 3 x 10 experiments
# Run 3x PPO discrete/continuous, A2C discrete/continuous, SAC


# SAC: continuous

# Continuous out of distribution
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1000 -t Summer 2019 -e Summer 2021 -m SAC -r 123
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1001 -t Summer 2019 -e Summer 2021 -m SAC -r 234
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1002 -t Summer 2019 -e Summer 2021 -m SAC -r 345

# Continuous in distribution
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1003 -t Summer 2021 -e Summer 2021 -m SAC -r 123
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1004 -t Summer 2021 -e Summer 2021 -m SAC -r 234
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1005 -t Summer 2021 -e Summer 2021 -m SAC -r 345


# PPO: discrete + continuous

# Continuous out of distribution
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1000 -t Summer 2019 -e Summer 2021 -m PPO -r 123
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1001 -t Summer 2019 -e Summer 2021 -m PPO -r 234
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1002 -t Summer 2019 -e Summer 2021 -m PPO -r 345

# Continuous in distribution
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1003 -t Summer 2021 -e Summer 2021 -m PPO -r 123
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1004 -t Summer 2021 -e Summer 2021 -m PPO -r 234
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1005 -t Summer 2021 -e Summer 2021 -m PPO -r 345

# Discrete out of distribution
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1006 -t Summer 2019 -e Summer 2021 -m PPO -d -r 123
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1007 -t Summer 2019 -e Summer 2021 -m PPO -d -r 234
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1008 -t Summer 2019 -e Summer 2021 -m PPO -d -r 345

# Discrete in distribution
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1009 -t Summer 2021 -e Summer 2021 -m PPO -d -r 123
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1010 -t Summer 2021 -e Summer 2021 -m PPO -d -r 234
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1011 -t Summer 2021 -e Summer 2021 -m PPO -d -r 345


# A2C: discrete + continuous

# Continuous out of distribution
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1000 -t Summer 2019 -e Summer 2021 -m A2C -r 123
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1001 -t Summer 2019 -e Summer 2021 -m A2C -r 234
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1002 -t Summer 2019 -e Summer 2021 -m A2C -r 345

# Continuous in distribution
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1003 -t Summer 2021 -e Summer 2021 -m A2C -r 123
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1004 -t Summer 2021 -e Summer 2021 -m A2C -r 234
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1005 -t Summer 2021 -e Summer 2021 -m A2C -r 345

# Discrete out of distribution
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1006 -t Summer 2019 -e Summer 2021 -m A2C -d -r 123
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1007 -t Summer 2019 -e Summer 2021 -m A2C -d -r 234
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1008 -t Summer 2019 -e Summer 2021 -m A2C -d -r 345

# Discrete in distribution
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1009 -t Summer 2021 -e Summer 2021 -m A2C -d -r 123
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1010 -t Summer 2021 -e Summer 2021 -m A2C -d -r 234
python -m sustaingym.scripts.evcharging.train_stable_baselines -n 1011 -t Summer 2021 -e Summer 2021 -m A2C -d -r 345
