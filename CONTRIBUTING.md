## Development Guide

1. Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html).
2. Create conda environment. Replace `XX` below with the name of the SustainGym environment you want to work on.
    ```bash
    conda env update --file env_XX.yml --prune
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


## Unit Tests

First, set your terminal directory to this repo's root directory. Next, make sure you have activated the appropriate conda environment for the SustainGym environment you want to test (e.g., `conda activate sustaingym_ev`). Finally, run the unit tests for the desired SustainGym environment:

```bash
python -m unittest -v tests/test_evcharging.py
```

## Building PyPI Package

```bash
# create a conda environment with appropriate build tools
conda env update --file env_build.yml --prune
conda activate build

# remove generated pkl files from CogenEnv
rm sustaingym/data/cogen/ambients_data/*.pkl

# clean cached build files
rm -r sustaingym.egg-info

# build source dist (.sdist) and wheel (.whl) files
python -m build

# upload built files to PyPI
twine upload dist/*
```
