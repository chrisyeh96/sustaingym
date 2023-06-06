# TO RUN, copy-paste in terminal:
# ./examples/evcharging/run_train_rllib_lr5e-6.sh

# DONE

# PPO Single-agent continuous
# python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-6


# --- SAC (does not support multidiscrete) ---

# Single-agent continuous
python -m examples.evcharging.train_rllib -a sac -t "Summer 2021" -s caltech -r 123 --lr 5e-6

# Multi-agent continuous
# python -m examples.evcharging.train_rllib -a sac -t "Summer 2021" -s caltech -r 123 --lr 5e-6 -m

# Multi-agent discrete
# python -m examples.evcharging.train_rllib -a sac -t "Summer 2021" -s caltech -r 123 --lr 5e-6 -m -d


# --- PPO ---

# Single-agent discrete
python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-6 -d

# Multi-agent continuous
# python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-6 -m

# Multi-agent discrete
# python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-6 -m -d
