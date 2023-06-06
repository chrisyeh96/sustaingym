#!/usr/bin/env sh

start="bash -i -c \"conda activate sustaingym && "
end="; bash\""

sessionname=congestedEMtraining

# 1st window will start ray server
cmd="ray start --head"
tmux new-session -d -s $sessionname -n main -d "$start $cmd $end"

echo "sleeping for 20s so that `ray start` finishes running"
sleep 20

# PPO Models
cmd="python3 train_rllib.py -m 7 -i -a ppo -l 5e-04 -o examples/interm_results"
tmux new-window -t $sessionname:1 -n sum_ppo_5e-4 "$start $cmd $end"
cmd="python3 train_rllib.py -m 7 -i -a ppo -l 5e-05 -o examples/interm_results"
tmux new-window -t $sessionname:2 -n sum_ppo_5e-5 "$start $cmd $end"
cmd="python3 train_rllib.py -m 7 -i -a ppo -l 5e-06 -o examples/interm_results"
tmux new-window -t $sessionname:3 -n sum_ppo_5e-6 "$start $cmd $end"
cmd="python3 train_rllib.py -m 12 -i -a ppo -l 5e-04 -o examples/interm_results"
tmux new-window -t $sessionname:1 -n wint_ppo_5e-4 "$start $cmd $end"
cmd="python3 train_rllib.py -m 12 -i -a ppo -l 5e-05 -o examples/interm_results"
tmux new-window -t $sessionname:2 -n wint_ppo_5e-5 "$start $cmd $end"
cmd="python3 train_rllib.py -m 12 -i -a ppo -l 5e-06 -o examples/interm_results"
tmux new-window -t $sessionname:3 -n wint_ppo_5e-6 "$start $cmd $end"

# SAC Models
cmd="python3 train_rllib.py -m 7 -i -a sac -l 5e-04 -o examples/interm_results"
tmux new-window -t $sessionname:1 -n sum_sac_5e-4 "$start $cmd $end"
cmd="python3 train_rllib.py -m 7 -i -a sac -l 5e-05 -o examples/interm_results"
tmux new-window -t $sessionname:2 -n sum_sac_5e-5 "$start $cmd $end"
cmd="python3 train_rllib.py -m 7 -i -a sac -l 5e-06 -o examples/interm_results"
tmux new-window -t $sessionname:3 -n sum_sac_5e-6 "$start $cmd $end"
cmd="python3 train_rllib.py -m 12 -i -a sac -l 5e-04 -o examples/interm_results"
tmux new-window -t $sessionname:1 -n wint_sac_5e-4 "$start $cmd $end"
cmd="python3 train_rllib.py -m 12 -i -a sac -l 5e-05 -o examples/interm_results"
tmux new-window -t $sessionname:2 -n wint_sac_5e-5 "$start $cmd $end"
cmd="python3 train_rllib.py -m 12 -i -a sac -l 5e-06 -o examples/interm_results"
tmux new-window -t $sessionname:3 -n wint_sac_5e-6 "$start $cmd $end"

# DQN Models
cmd="python3 train_rllib.py -m 7 -d -i -a sac -l 5e-04 -o examples/interm_results"
tmux new-window -t $sessionname:1 -n sum_dqn_5e-4 "$start $cmd $end"
cmd="python3 train_rllib.py -m 7 -d -i -a sac -l 5e-05 -o examples/interm_results"
tmux new-window -t $sessionname:2 -n sum_dqn_5e-5 "$start $cmd $end"
cmd="python3 train_rllib.py -m 7 -d -i -a sac -l 5e-06 -o examples/interm_results"
tmux new-window -t $sessionname:3 -n sum_dqn_5e-6 "$start $cmd $end"
cmd="python3 train_rllib.py -m 12 -d -i -a sac -l 5e-04 -o examples/interm_results"
tmux new-window -t $sessionname:1 -n wint_dqn_5e-4 "$start $cmd $end"
cmd="python3 train_rllib.py -m 12 -d -i -a sac -l 5e-05 -o examples/interm_results"
tmux new-window -t $sessionname:2 -n wint_dqn_5e-5 "$start $cmd $end"
cmd="python3 train_rllib.py -m 12 -d -i -a sac -l 5e-06 -o examples/interm_results"
tmux new-window -t $sessionname:3 -n wint_dqn_5e-6 "$start $cmd $end"


tmux attach -t $sessionname
