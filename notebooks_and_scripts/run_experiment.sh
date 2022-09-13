# 12 experiments total

# PPO
python -m notebooks_and_scripts.train --exp 301 --seed 123 --project_action_in_env True --model ppo --site caltech --train_period summer2019 --test_periods summer2021
python -m notebooks_and_scripts.train --exp 302 --seed 246 --project_action_in_env True --model ppo --site caltech --train_period summer2019 --test_periods summer2021
python -m notebooks_and_scripts.train --exp 303 --seed 369 --project_action_in_env True --model ppo --site caltech --train_period summer2019 --test_periods summer2021

python -m notebooks_and_scripts.train --exp 311 --seed 123 --project_action_in_env False --model ppo --site caltech --train_period summer2021 --test_periods summer2021
python -m notebooks_and_scripts.train --exp 312 --seed 246 --project_action_in_env False --model ppo --site caltech --train_period summer2021 --test_periods summer2021
python -m notebooks_and_scripts.train --exp 313 --seed 369 --project_action_in_env False --model ppo --site caltech --train_period summer2021 --test_periods summer2021

# A2C
python -m notebooks_and_scripts.train --exp 401 --seed 123 --project_action_in_env True --model a2c --site caltech --train_period summer2019 --test_periods summer2021
python -m notebooks_and_scripts.train --exp 402 --seed 246 --project_action_in_env True --model a2c --site caltech --train_period summer2019 --test_periods summer2021
python -m notebooks_and_scripts.train --exp 403 --seed 369 --project_action_in_env True --model a2c --site caltech --train_period summer2019 --test_periods summer2021

python -m notebooks_and_scripts.train --exp 411 --seed 123 --project_action_in_env False --model a2c --site caltech --train_period summer2021 --test_periods summer2021
python -m notebooks_and_scripts.train --exp 412 --seed 246 --project_action_in_env False --model a2c --site caltech --train_period summer2021 --test_periods summer2021
python -m notebooks_and_scripts.train --exp 413 --seed 369 --project_action_in_env False --model a2c --site caltech --train_period summer2021 --test_periods summer2021
