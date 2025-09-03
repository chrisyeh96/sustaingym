#!/usr/bin/env sh

start="bash -i -c \"conda activate sustaingym_em && "
end="; bash\""

sessionname=congestedEMtraining

# 1st window will start ray server
cmd="ray start --head"
tmux new-session -d -s $sessionname -n main -d "$start $cmd $end"

echo "sleeping for 20s so that 'ray start' finishes running"
sleep 20

# # PPO Models
# cmd="python -m examples.train_rllib -m 7 -i -a ppo -l 5e-04 -o examples/interm_results"
# tmux new-window -t $sessionname:1 -n sum_ppo_5e-4 "$start $cmd $end"
# cmd="python -m examples.train_rllib -m 7 -i -a ppo -l 5e-05 -o examples/interm_results"
# tmux new-window -t $sessionname:2 -n sum_ppo_5e-5 "$start $cmd $end"
# cmd="python -m examples.train_rllib -m 7 -i -a ppo -l 5e-06 -o examples/interm_results"
# tmux new-window -t $sessionname:3 -n sum_ppo_5e-6 "$start $cmd $end"
# cmd="python -m examples.train_rllib -m 12 -i -a ppo -l 5e-04 -o examples/interm_results"
# tmux new-window -t $sessionname:4 -n wint_ppo_5e-4 "$start $cmd $end"
# cmd="python -m examples.train_rllib -m 12 -i -a ppo -l 5e-05 -o examples/interm_results"
# tmux new-window -t $sessionname:5 -n wint_ppo_5e-5 "$start $cmd $end"
# cmd="python -m examples.train_rllib -m 12 -i -a ppo -l 5e-06 -o examples/interm_results"
# tmux new-window -t $sessionname:6 -n wint_ppo_5e-6 "$start $cmd $end"

# # SAC Models
# cmd="python -m examples.train_rllib -m 7 -i -a sac -l 5e-04 -o examples/interm_results"
# tmux new-window -t $sessionname:7 -n sum_sac_5e-4 "$start $cmd $end"
# cmd="python -m examples.train_rllib -m 7 -i -a sac -l 5e-05 -o examples/interm_results"
# tmux new-window -t $sessionname:8 -n sum_sac_5e-5 "$start $cmd $end"
# cmd="python -m examples.train_rllib -m 7 -i -a sac -l 5e-06 -o examples/interm_results"
# tmux new-window -t $sessionname:9 -n sum_sac_5e-6 "$start $cmd $end"
# cmd="python -m examples.train_rllib -m 12 -i -a sac -l 5e-04 -o examples/interm_results"
# tmux new-window -t $sessionname:10 -n wint_sac_5e-4 "$start $cmd $end"
# cmd="python -m examples.train_rllib -m 12 -i -a sac -l 5e-05 -o examples/interm_results"
# tmux new-window -t $sessionname:11 -n wint_sac_5e-5 "$start $cmd $end"
# cmd="python -m examples.train_rllib -m 12 -i -a sac -l 5e-06 -o examples/interm_results"
# tmux new-window -t $sessionname:12 -n wint_sac_5e-6 "$start $cmd $end"

# DQN Models
cmd="python -m examples.train_rllib -m 7 -d -i -a dqn -l 5e-04 -o examples/interm_results"
tmux new-window -t $sessionname:13 -n sum_dqn_5e-4 "$start $cmd $end"
cmd="python -m examples.train_rllib -m 7 -d -i -a dqn -l 5e-05 -o examples/interm_results"
tmux new-window -t $sessionname:14 -n sum_dqn_5e-5 "$start $cmd $end"
cmd="python -m examples.train_rllib -m 7 -d -i -a dqn -l 5e-06 -o examples/interm_results"
tmux new-window -t $sessionname:15 -n sum_dqn_5e-6 "$start $cmd $end"
cmd="python -m examples.train_rllib -m 12 -d -i -a dqn -l 5e-04 -o examples/interm_results"
tmux new-window -t $sessionname:16 -n wint_dqn_5e-4 "$start $cmd $end"
cmd="python -m examples.train_rllib -m 12 -d -i -a dqn -l 5e-05 -o examples/interm_results"
tmux new-window -t $sessionname:17 -n wint_dqn_5e-5 "$start $cmd $end"
cmd="python -m examples.train_rllib -m 12 -d -i -a dqn -l 5e-06 -o examples/interm_results"
tmux new-window -t $sessionname:18 -n wint_dqn_5e-6 "$start $cmd $end"


tmux attach -t $sessionname
