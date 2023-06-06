# TO RUN, copy-paste in terminal:
# ./examples/evcharging/run_train_rllib_lr5e-4.sh


# --- PPO ---

# Single-agent continuous
python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-4

# Single-agent discrete
python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-4 -d

# NOT WORKING: Multi-agent continuous
# python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-4 -m

# NOT WORKING: Multi-agent discrete
# python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-4 -m -d


# --- SAC ---

# Single-agent continuous
python -m examples.evcharging.train_rllib -a sac -t "Summer 2021" -s caltech -r 123 --lr 5e-4

# Single-agent discrete
python -m examples.evcharging.train_rllib -a sac -t "Summer 2021" -s caltech -r 123 --lr 5e-4 -d

# NOT WORKING: Multi-agent continuous
# python -m examples.evcharging.train_rllib -a sac -t "Summer 2021" -s caltech -r 123 --lr 5e-4 -m

# NOT WORKING: Multi-agent discrete
# python -m examples.evcharging.train_rllib -a sac -t "Summer 2021" -s caltech -r 123 --lr 5e-4 -m -d
