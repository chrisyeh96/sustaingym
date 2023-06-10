#!/usr/bin/env sh
export RAY_PYTHON=/opt/conda/envs/pytorch/bin/python

start="bash -i -c \"conda activate pytorch && "
end="; bash\""

sessionname=COGENBASELINE

# 1st window will start ray server
cmd="ray start --head"
tmux new-session -d -s $sessionname -n main -d "$start $cmd $end"

echo "sleeping for 20s so that ray start finishes running"
sleep 20

cmd="python -m examples.cogen.train_rllib -a rand -r 123 --lr 5e-4 --rm 300"
tmux new-window -t $sessionname:1 -n rand_5e-4 "$start $cmd $end"
# cmd="python -m examples.cogen.train_rllib -a ppo -r 123 --lr 5e-5 --rm 300"
# tmux new-window -t $sessionname:2 -n ppo_5e-5 "$start $cmd $end"
# cmd="python -m examples.cogen.train_rllib -a ppo -r 123 --lr 5e-6 --rm 300"
# tmux new-window -t $sessionname:3 -n ppo_5e-6 "$start $cmd $end"

# cmd="python -m examples.cogen.train_rllib -a ppo -r 456 --lr 5e-4 --rm 300"
# tmux new-window -t $sessionname:4 -n ppo_5e-4 "$start $cmd $end"
# cmd="python -m examples.cogen.train_rllib -a ppo -r 456 --lr 5e-5 --rm 300"
# tmux new-window -t $sessionname:5 -n ppo_5e-5 "$start $cmd $end"
# cmd="python -m examples.cogen.train_rllib -a ppo -r 456 --lr 5e-6 --rm 300"
# tmux new-window -t $sessionname:6 -n ppo_5e-6 "$start $cmd $end"

# cmd="python -m examples.cogen.train_rllib -a ppo -r 789 --lr 5e-4 --rm 300"
# tmux new-window -t $sessionname:7 -n ppo_5e-4 "$start $cmd $end"
# cmd="python -m examples.cogen.train_rllib -a ppo -r 789 --lr 5e-5 --rm 300"
# tmux new-window -t $sessionname:8 -n ppo_5e-5 "$start $cmd $end"
# cmd="python -m examples.cogen.train_rllib -a ppo -r 789 --lr 5e-6 --rm 300"
# tmux new-window -t $sessionname:9 -n ppo_5e-6 "$start $cmd $end"


tmux attach -t $sessionname
