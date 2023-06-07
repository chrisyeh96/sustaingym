# SustainGym: Reinforcement learning environments for sustainability applications

## Folder structure

```
examples/               # example code for running each environment
sustaingym/             # main Python package
    algorithms/
        {env}/          # per-env algorithms
    data/
        moer/           # marginal carbon emission rates
        {env}/          # per-env data files
    envs/
        {env}/          # per-env modules
```


## Development Guide

1. Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html).
2. Create conda environment.
    ```bash
    # for environment development
    conda env update --file env_norl.yml --prune

    # for RL development
    conda env update --file env.yml --prune
    ```

   If you are using RLLib with a GPU, you will also need to [configure TensorFlow for GPU](https://www.tensorflow.org/install/pip#4_gpu_setup):
    ```bash
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    ```

3. Make code modifications in a separate git branch
    ```bash
    git checkout -b new_feature
    ```
4. From repo root folder, run mypy type checker and fix any errors.
    ```bash
    mypy sustaingym
    ```
5. From repo root folder, run code linter and fix any linting errors.
    ```bash
    flake8 sustaingym
    ```
6. Commit changes in git and push.
7. Submit pull request on GitHub.
