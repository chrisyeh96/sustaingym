# SustainGym: Reinforcement learning environments for sustainability applications

The lack of standardized benchmarks for reinforcement learning (RL) in sustainability applications has made it difficult to both track progress on specific domains and identify bottlenecks for researchers to focus their efforts on. We present SustainGym, a suite of environments designed to test the performance of RL algorithms on realistic sustainability tasks. These environments highlight challenges in introducing RL to real-world sustainability tasks, including physical constraints and distribution shift.

[**Paper**](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2022/38/paper.pdf)
| [**Website**](https://chrisyeh96.github.io/sustaingym/)

Please see the [SustainGym website](https://chrisyeh96.github.io/sustaingym/) for a getting started guide.


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


## Citation

Please cite SustainGym as

> C. Yeh, V. Li, R. Datta, Y. Yue, and A. Wierman, "SustainGym: A Benchmark Suite of Reinforcement Learning for Sustainability Applications," in _NeurIPS 2022 Workshop on Tackling Climate Change with Machine Learning_, Dec. 2022. [Online]. Available: [https://www.climatechange.ai/papers/neurips2022/38](https://www.climatechange.ai/papers/neurips2022/38).

<details markdown="block">
<summary>BibTeX</summary>

```tex
@inproceedings{yeh2022sustaingym,
    title={SustainGym: A Benchmark Suite of Reinforcement Learning for Sustainability Applications},
    author={Yeh, Christopher and Li, Victor and Datta, Rajeev and Yue, Yisong and Wierman, Adam},
    booktitle={NeurIPS 2022 Workshop on Tackling Climate Change with Machine Learning},
    url={https://www.climatechange.ai/papers/neurips2022/38},
    year={2022},
    month={12}
}
```

</details>