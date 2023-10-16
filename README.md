# SustainGym: Reinforcement learning environments for sustainability applications

The lack of standardized benchmarks for reinforcement learning (RL) in sustainability applications has made it difficult to both track progress on specific domains and identify bottlenecks for researchers to focus their efforts on. We present SustainGym, a suite of environments designed to test the performance of RL algorithms on realistic sustainability tasks. These environments highlight challenges in introducing RL to real-world sustainability tasks, including physical constraints and distribution shift.

[**Paper**](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2022/38/paper.pdf)
| [**Website**](https://chrisyeh96.github.io/sustaingym/)

SustainGym contains both single-agent and multi-agent RL environments.
- Single-agent environments follow the [Gymnasium API](https://gymnasium.farama.org/) and are designed to be easily used with the [StableBaselines3](https://stable-baselines3.readthedocs.io/) and [Ray RLLib](https://docs.ray.io/en/latest/rllib/) libraries for training RL algorithms.
- Multi-agent environments follow the [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/) and are designed to be easily used with the [Ray RLLib](https://docs.ray.io/en/latest/rllib/) library for training multi-agent RL algorithms.

Please see the [SustainGym website](https://chrisyeh96.github.io/sustaingym/) for a getting started guide and complete documentation.


## Folder structure

```
docs/                   # website and documentation
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

## License

SustainGym is released under a [Creative Commons Attribution 4.0 International Public License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See the [LICENSE](LICENSE) file for the full terms.

## Citation

Please cite SustainGym as

> C. Yeh, V. Li, R. Datta, J. Arroyo, N. Christianson, C. Zhang, Y. Chen, M. Hosseini, A. Golmohammadi, Y. Shi, Y. Yue, and A. Wierman, "SustainGym: A Benchmark Suite of Reinforcement Learning for Sustainability Applications," in _Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track_, New Orleans, LA, USA, Dec. 2023. [Online]. Available: [https://openreview.net/forum?id=vZ9tA3o3hr](https://openreview.net/forum?id=vZ9tA3o3hr).

<details markdown="block">
<summary>BibTeX</summary>

```tex
@inproceedings{yeh2023sustaingym,
    title = {{SustainGym}: Reinforcement Learning Environments for Sustainable Energy Systems},
    author = {Yeh, Christopher and Li, Victor and Datta, Rajeev and Arroyo, Julio and Zhang, Chi and Chen, Yize and Hosseini, Mehdi and Golmohammadi, Azarang and Shi, Yuanyuan and Yue, Yisong and Wierman, Adam},
    year = 2023,
    month = 12,
    booktitle = {Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    address = {New Orleans, LA, USA},
    url = {https://openreview.net/forum?id=vZ9tA3o3hr}
}
```

</details>

An earlier version of this work was published as a workshop paper:

> C. Yeh, V. Li, R. Datta, Y. Yue, and A. Wierman, "SustainGym: A Benchmark Suite of Reinforcement Learning for Sustainability Applications," in _NeurIPS 2022 Workshop on Tackling Climate Change with Machine Learning_, Dec. 2022. [Online]. Available: [https://www.climatechange.ai/papers/neurips2022/38](https://www.climatechange.ai/papers/neurips2022/38).

<details markdown="block">
<summary>BibTeX</summary>

```tex
@inproceedings{yeh2022sustaingym,
    title = {{SustainGym}: A Benchmark Suite of Reinforcement Learning for Sustainability Applications},
    author = {Yeh, Christopher and Li, Victor and Datta, Rajeev and Yue, Yisong and Wierman, Adam},
    year = 2022,
    month = 12,
    booktitle = {NeurIPS 2022 Workshop on Tackling Climate Change with Machine Learning},
    address = {New Orleans, LA, USA},
    url = {https://www.climatechange.ai/papers/neurips2022/38}
}
```

</details>