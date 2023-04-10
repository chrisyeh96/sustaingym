# Datacenter Simulator (Gym API)

This is a datacenter simulator that implements the Gym API. It serves as an environment to train reinforcement learning agents.

### One-time data preparation

1. You will need to obtain a data sample from the original dataset provided by Google: https://github.com/google/cluster-data . You can use the SQL commands we used following the documentation in this repo at sustaingym/data/datacenter/SQL_docs.md

2. After obtaining your data sample, you can format it with the script at sustaingym/data/datacenter/preproc/daily_split.py

### Example usage

After following the one-time data preparation steps, you can interact with the Gym environment following the example:

`python3 sustaingym/algorithms/datacenter/example.py`
