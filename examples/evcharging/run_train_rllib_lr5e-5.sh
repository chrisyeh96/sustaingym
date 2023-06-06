# TO RUN, copy-paste in terminal:
# ./examples/evcharging/run_train_rllib_lr5e-5.sh


# DONE
# Single-agent continuous
# python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-5



# --- SAC ---

# Single-agent discrete
python -m examples.evcharging.train_rllib -a sac -t "Summer 2021" -s caltech -r 123 --lr 5e-5 -d

# Single-agent continuous
python -m examples.evcharging.train_rllib -a sac -t "Summer 2021" -s caltech -r 123 --lr 5e-5

# NOT WORKING: Multi-agent continuous
# python -m examples.evcharging.train_rllib -a sac -t "Summer 2021" -s caltech -r 123 --lr 5e-5 -m

# NOT WORKING: Multi-agent discrete
# python -m examples.evcharging.train_rllib -a sac -t "Summer 2021" -s caltech -r 123 --lr 5e-5 -m -d



# --- PPO ---

# Single-agent discrete
python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-5 -d

# NOT WORKING: Multi-agent continuous
# python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-5 -m

# NOT WORKING: Multi-agent discrete
# python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-5 -m -d
