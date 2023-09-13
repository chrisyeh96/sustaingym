---
hide-toc: true
---

# SustainGym

<p class="fs-6 fw-300">
A suite of environments designed to test the performance of RL algorithms on realistic sustainability tasks.
</p>

The lack of standardized benchmarks for reinforcement learning (RL) in sustainability applications has made it difficult to both track progress on specific domains and identify bottlenecks for researchers to focus their efforts on. We present **SustainGym**, a suite of environments designed to test the performance of RL algorithms on realistic sustainability tasks. These environments highlight challenges in introducing RL to real-world sustainability tasks, including physical constraints and distribution shift.

<p>
    <a href="https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2022/38/paper.pdf" class="btn btn-blue fs-5 mb-4 mb-md-0 mr-2">Read the Paper</a>
    <a href="https://github.com/chrisyeh96/sustaingym/" class="btn fs-5 mb-4 mb-md-0">View it on GitHub</a>
</p>

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
