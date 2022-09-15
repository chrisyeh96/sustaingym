# 12 experiments total

# PPO
python -m notebooks_and_scripts.train --exp 101 --seed 123 --action_type discrete --model ppo --site caltech --train_period summer2019 --test_period summer2019 fall2019 spring2020 summer2021
python -m notebooks_and_scripts.train --exp 102 --seed 246 --action_type discrete --model ppo --site caltech --train_period summer2019 --test_period summer2019 fall2019 spring2020 summer2021
python -m notebooks_and_scripts.train --exp 103 --seed 369 --action_type discrete --model ppo --site caltech --train_period summer2019 --test_period summer2019 fall2019 spring2020 summer2021

# python -m notebooks_and_scripts.train --exp 111 --seed 123 --action_type discrete --model ppo --site caltech --train_period spring2020 --test_period spring2020
# python -m notebooks_and_scripts.train --exp 112 --seed 246 --action_type discrete --model ppo --site caltech --train_period spring2020 --test_period spring2020
# python -m notebooks_and_scripts.train --exp 113 --seed 369 --action_type discrete --model ppo --site caltech --train_period spring2020 --test_period spring2020

python -m notebooks_and_scripts.train --exp 121 --seed 123 --action_type discrete --model ppo --site caltech --train_period summer2021 --test_period summer2021
python -m notebooks_and_scripts.train --exp 122 --seed 246 --action_type discrete --model ppo --site caltech --train_period summer2021 --test_period summer2021
python -m notebooks_and_scripts.train --exp 123 --seed 369 --action_type discrete --model ppo --site caltech --train_period summer2021 --test_period summer2021


# A2C
python -m notebooks_and_scripts.train --exp 201 --seed 123 --action_type discrete --model a2c --site caltech --train_period summer2019 --test_period summer2019 fall2019 spring2020 summer2021
python -m notebooks_and_scripts.train --exp 202 --seed 246 --action_type discrete --model a2c --site caltech --train_period summer2019 --test_period summer2019 fall2019 spring2020 summer2021
python -m notebooks_and_scripts.train --exp 203 --seed 369 --action_type discrete --model a2c --site caltech --train_period summer2019 --test_period summer2019 fall2019 spring2020 summer2021

# python -m notebooks_and_scripts.train --exp 211 --seed 123 --action_type discrete --model a2c --site caltech --train_period spring2020 --test_period spring2020
# python -m notebooks_and_scripts.train --exp 212 --seed 246 --action_type discrete --model a2c --site caltech --train_period spring2020 --test_period spring2020
# python -m notebooks_and_scripts.train --exp 213 --seed 369 --action_type discrete --model a2c --site caltech --train_period spring2020 --test_period spring2020

python -m notebooks_and_scripts.train --exp 221 --seed 123 --action_type discrete --model a2c --site caltech --train_period summer2021 --test_period summer2021
python -m notebooks_and_scripts.train --exp 222 --seed 246 --action_type discrete --model a2c --site caltech --train_period summer2021 --test_period summer2021
python -m notebooks_and_scripts.train --exp 223 --seed 369 --action_type discrete --model a2c --site caltech --train_period summer2021 --test_period summer2021

# Evaluate baselines
python -m notebooks_and_scripts.run_baselines
