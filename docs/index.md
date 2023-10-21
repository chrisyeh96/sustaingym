---
hide-toc: true
---

# SustainGym

<p class="fs-6 fw-300">
A suite of environments designed to test the performance of RL algorithms on realistic sustainability tasks.
</p>

The lack of standardized benchmarks for reinforcement learning (RL) in sustainability applications has made it difficult to both track progress on specific domains and identify bottlenecks for researchers to focus their efforts on. We present **SustainGym**, a suite of environments designed to test the performance of RL algorithms on realistic sustainability tasks. These environments highlight challenges in introducing RL to real-world sustainability tasks, including physical constraints and distribution shift.

<p>
    <a href="https://drive.google.com/file/d/1wrLGu2FCVOT_BvtDoudsz05zFG89r7dI/view?usp=drive_link" class="btn btn-blue fs-5 mb-4 mb-md-0 mr-2">Read the Paper</a>
    <a href="https://github.com/chrisyeh96/sustaingym/" class="btn fs-5 mb-4 mb-md-0">View it on GitHub</a>
</p>

SustainGym is released under a [Creative Commons Attribution 4.0 International Public License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Please cite SustainGym as

> C. Yeh, V. Li, R. Datta, J. Arroyo, N. Christianson, C. Zhang, Y. Chen, M. Hosseini, A. Golmohammadi, Y. Shi, Y. Yue, and A. Wierman, "SustainGym: A Benchmark Suite of Reinforcement Learning for Sustainability Applications," in _Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track_, New Orleans, LA, USA, Dec. 2023. [Online]. Available: [https://openreview.net/forum?id=vZ9tA3o3hr](https://openreview.net/forum?id=vZ9tA3o3hr).

<details markdown="block">
<summary>BibTeX</summary>

```tex
@inproceedings{yeh2023sustaingym,
    title = {{SustainGym}: {Reinforcement} Learning Environments for Sustainable Energy Systems},
    author = {Yeh, Christopher and Li, Victor and Datta, Rajeev and Arroyo, Julio and Christianson, Nicolas and Zhang, Chi and Chen, Yize and Hosseini, Mehdi and Golmohammadi, Azarang and Shi, Yuanyuan and Yue, Yisong and Wierman, Adam},
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

```{toctree}
:hidden:
:maxdepth: 1

Home <self>
team
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Environments

evchargingenv
electricitymarketenv
datacenterenv
cogenenv
buildingenv
```

```{toctree}
:hidden:
:caption: API Reference

api/sustaingym/algorithms/index
api/sustaingym/data/index
api/sustaingym/envs/index
genindex
modindex
```
