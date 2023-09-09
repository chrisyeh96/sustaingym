# SustainGym: Reinforcement learning environments for sustainability applications

The lack of standardized benchmarks for reinforcement learning (RL) in sustainability applications has made it difficult to both track progress on specific domains and identify bottlenecks for researchers to focus their efforts on. We present SustainGym, a suite of environments designed to test the performance of RL algorithms on realistic sustainability tasks. These environments highlight challenges in introducing RL to real-world sustainability tasks, including physical constraints and distribution shift.

[**Paper**](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2022/38/paper.pdf)
| [**Website**](https://chrisyeh96.github.io/sustaingym/)

SustainGym contains both single-agent and multi-agent RL environments.
- Single-agent environments follow the [Gymnasium API](https://gymnasium.farama.org/) and are designed to be easily used with the [StableBaselines3](https://stable-baselines3.readthedocs.io/) and [Ray RLLib](https://docs.ray.io/en/latest/rllib/) libraries for training RL algorithms.
- Multi-agent environments follow the [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/) and are designed to be easily used with the [Ray RLLib](https://docs.ray.io/en/latest/rllib/) library for training multi-agent RL algorithms.

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
tests/                  # unit tests
```

## Contributing

If you would like to add a new environment, propose bug fixes, or otherwise contribute to SustainGym, please see the [Contributing Guide](CONTRIBUTING.md).


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