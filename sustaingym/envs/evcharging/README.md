## EV Charging

We downloaded EV charging data from the [Adaptive Charging Network website](https://ev.caltech.edu/dataset) on September 16, 2022, using the command:

```bash
python -m sustaingym.envs.evcharging.utils
```

The default Gaussian Mixture Models (GMMs) were then trained using train_gmm_model.py:

```bash
python -m sustaingym.envs.evcharging.train_gmm_model --site caltech --gmm_n_components 30 --date_range 2019-05-01 2019-08-31 2019-09-01 2019-12-31 2020-02-01 2020-05-31 2021-05-01 2021-08-31
python -m sustaingym.envs.evcharging.train_gmm_model --site jpl --gmm_n_components 30 --date_range 2019-05-01 2019-08-31 2019-09-01 2019-12-31 2020-02-01 2020-05-31 2021-05-01 2021-08-31
```

The charging data is made possible from a close collaboration between Caltech and PowerFlex, and the gym currently supports the Caltech (54 charging stations) and JPL (52 charging stations) sites.
