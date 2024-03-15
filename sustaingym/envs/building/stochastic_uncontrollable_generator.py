from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import scipy


class StochasticUncontrollableGenerator:
    """
    A generator class for the uncontrollable features in BuildingEnv.

    Args:
        num_episodes: The number of episodes in the provided observation
            data. If given, instance will treat each row in the observation
            data as a separate raw dataset to split into seasons and fit
            distributions to. If None, will use the provided raw data as a
            single episode.
    """

    def __init__(self, num_episodes: int | None = None):
        self.episodes = num_episodes

        self.observations = [[]]

        self.summer_observations = []
        self.winter_observations = []

        self.summer_dists = None
        self.winter_dists = None

        self.block_size = 0

    def split_observations_into_seasons(
        self, observation_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits observation data into summer and winter seasons.

        Args:
            observation_data: The collected observation data. Shape
                is (num_episodes x len_of_episode, num_features).

        Returns:
            summer_observations: Summer ambient features. Shape is
                (len_of_season x num_episodes, num_features).
            winter_observations: Winter ambient features. Shape is
                (len_of_season x num_episodes, num_features).

        Raises:
            ValueError if no observation_data is given and the class instance
                has no observations stored.
        """
        if observation_data is None and len(self.observations) == 0:
            raise ValueError("`observation_data` cannot be None")

        self.summer_observations = []
        self.winter_observations = []

        if self.episodes is not None:
            for i in range(self.episodes):
                this_observation_block = observation_data[i]
                this_n = len(this_observation_block)

                # January only for winter distribution
                this_winter = this_observation_block[: this_n // 12]
                # July only for summer distribution
                this_summer = this_observation_block[
                    this_n * 2 // 12 : this_n * 3 // 12
                ]

                if i == 0:
                    self.summer_observations = this_summer
                    self.winter_observations = this_winter
                else:
                    self.summer_observations = np.vstack(
                        (self.summer_observations, this_summer)
                    )
                    self.winter_observations = np.vstack(
                        (self.winter_observations, this_winter)
                    )
        else:
            num_observations = observation_data.shape[0]
            self.summer_observations = observation_data[
                num_observations // 4 : 3 * num_observations // 4, :
            ]
            self.winter_observations = np.vstack(
                (
                    observation_data[: num_observations // 4, :],
                    observation_data[3 * num_observations // 4 :, :],
                )
            )

        # (len_of_season x num_episodes, dim_of_obs_vector)
        self.summer_observations = np.array(self.summer_observations)
        self.winter_observations = np.array(self.winter_observations)

        return self.summer_observations, self.winter_observations

    def get_empirical_dist(
        self,
        season: str | None = None,
        this_season_observations: np.ndarray | None = None,
        block_size: int = 100,
    ) -> list[scipy.stats.rv_continuous]:
        """
        Fits a multivariate normal distribution to each ambient feature.

        Args:
            season: The desired season. Can be `None` if this_season_observations is
                not None. Otherwise, can be `summer` or `winter` if data has been generated
                and split into seasons through split_observations_into_seasons.
            this_season_observations: The observation data for this season. Can be
                `None` if user has generated and split data.
            block_size: Desired block size for each ambient feature vector.
                In hours.

        Returns:
            empirical_dists: The empirical distributions for each of the ambient
                features.

        Raises:
            ValueError if neither season nor this_season_observations is specified OR
                if the season is given but it is not "summer" or "winter"
        """
        if season is None and this_season_observations is None:
            raise ValueError(
                "Either season or this_season_observations must be specified."
            )
        if season is not None and this_season_observations is None:
            if season == "summer":
                this_season_observations = self.summer_observations
            elif season == "winter":
                this_season_observations = self.winter_observations
            else:
                raise ValueError("Season must be either summer or winter.")

        this_season_observations = np.array(this_season_observations)
        num_obs, num_features = this_season_observations.shape
        self.block_size = block_size

        mu_vectors = []
        cov_matrices = []
        for i in range(num_features):
            this_col = this_season_observations[:, i]
            this_col = this_col[
                : int((num_obs // block_size) * block_size)
            ]  # truncates data to be divisible by block_size
            reshaped_obs = this_col.reshape(
                block_size, num_obs // block_size, order="F"
            )
            this_mu_vector = np.mean(reshaped_obs, axis=1)
            this_cov_mat = np.cov(reshaped_obs)
            mu_vectors.append(this_mu_vector)
            cov_matrices.append(this_cov_mat)

        empirical_dists = []
        for i in range(len(mu_vectors)):
            this_dist = scipy.stats.multivariate_normal(
                mean=mu_vectors[i], cov=cov_matrices[i], allow_singular=True
            )
            empirical_dists.append(this_dist)

        if season == "summer":
            self.summer_dists = empirical_dists
        elif season == "winter":
            self.winter_dists = empirical_dists

        return empirical_dists

    def draw_samples_from_dist(
        self,
        num_samples: int,
        summer_percentage: float,
        dists: Iterable[scipy.stats.rv_continuous] | None = None,
        block_size: int | None = None,
    ) -> np.ndarray:
        """
        Draw vector samples from fitted multivariate Gaussian.

        Args:
            num_samples: The number of desired samples.
            summer_percentage: The weight of the generated observations to
                be given to those generated from the summer distribution.
            dists: Iterable of the empirical distributions. Can be `None` if
                instance has generated empirical distributions through
                get_empirical_dist.

        Returns:
            samples: The samples generated from the fitted distributions.
                Shape is (num_samples x block_size, num_obs_features).

        Raises:
            ValueError if `summer_percentage` is not between 0 and 1 or if
                either the summer or winter distributions aren't available to
                draw samples from.
        """
        if summer_percentage > 1 or summer_percentage < 0:
            raise ValueError("`summer_percentage` must be between 0 and 1")
        if dists is None:
            if self.summer_dists is None or self.winter_dists is None:
                raise ValueError("No dists available; call `get_empicial_dist` first")
            else:
                dists = [self.summer_dists, self.winter_dists]

        if block_size is None:
            block_size = self.block_size

        num_dists = len(dists[0])
        num_blocks = num_samples // block_size + 1

        season_obs = np.zeros((num_samples, num_dists))

        # blending dists for summer and winter
        blended_dists = []
        for dist_idx in range(num_dists):
            summer_dist = dists[0][dist_idx]
            winter_dist = dists[1][dist_idx]

            blended_mean = (
                summer_dist.mean * summer_percentage
                + (1 - summer_percentage) * winter_dist.mean
            )
            blended_cov = (
                summer_dist.cov * summer_percentage
                + (1 - summer_percentage) * winter_dist.cov
            )
            blended_dist = scipy.stats.multivariate_normal(
                mean=blended_mean, cov=blended_cov, allow_singular=True
            )
            blended_dists.append(blended_dist)

        samples = []
        for i in range(num_dists):
            this_dist = blended_dists[i]
            this_samples = this_dist.rvs(size=num_blocks)

            this_samples = this_samples.reshape(-1, 1)
            samples.append(this_samples)
        samples = np.stack(samples, axis=1)
        samples = samples[:num_samples, :].squeeze()

        season_obs += samples

        return season_obs
