#!/usr/bin/env sh

start="bash -i -c \"conda activate sustaingym && "
end="; bash\""

sessionname=YOURSESSIONNAME

# 1st window will start ray server
cmd="ray start --head"
tmux new-session -d -s $sessionname -n main -d "$start $cmd $end"

echo "sleeping for 20s so that `ray start` finishes running"
sleep 20

cmd="python -m examples.YOURENV.train_rllib -a ppo -r 123 --lr 5e-4"
tmux new-window -t $sessionname:1 -n ppo_5e-4 "$start $cmd $end"
cmd="python -m examples.YOURENV.train_rllib -a ppo -r 123 --lr 5e-5"
tmux new-window -t $sessionname:2 -n ppo_5e-5 "$start $cmd $end"
cmd="python -m examples.YOURENV.train_rllib -a ppo -r 123 --lr 5e-6"
tmux new-window -t $sessionname:3 -n ppo_5e-6 "$start $cmd $end"

cmd="python -m examples.YOURENV.train_rllib -a sac -r 123 --lr 1e-2"
tmux new-window -t $sessionname:4 -n sac "$start $cmd $end"
cmd="python 0-m examples.YOURENV.train_rllib -a sac -r 123 --lr 1e-3"
tmux new-window -t $sessionname:5 -n sac "$start $cmd $end"
cmd="python -m examples.YOURENV.train_rllib -a sac -r 123 --lr 1e-4"
tmux new-window -t $sessionname:6 -n sac "$start $cmd $end"

tmux attach -t $sessionname
