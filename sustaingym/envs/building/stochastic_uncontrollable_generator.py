from typing import Optional, Iterable

import numpy as np
import pandas as pd
import scipy

class StochasticUncontrollableGenerator():
    def __init__(self, 
                 building_env = None, 
                 num_episodes: int = None):
        """
        Initializes a generator class for the uncontrollable features in BuildingEnv.

        Args:
            building_env: The instance of the BuildingEnv environment.
            num_episodes: The number of episodes over which to collect observations
                from the environment.
        """
        self.env = building_env
        self.episodes = num_episodes

        self.observations = [[]]

        self.summer_observations = []
        self.winter_observations = []

        self.summer_dists = None
        self.winter_dists = None
        
        self.block_size = 0

    def split_observations_into_seasons(self, observation_data: Optional[np.array] = None):
        """
        Splits observation data into summer and winter seasons.

        Args:
        observation_data: The collected observation data. Can be `None` if 
            collect_random_observations has been called.
        
        Returns:
            summer_observations: Summer ambient features.
            winter_observations: Winter ambient features.

        Raises:
            ValueError if no observation_data is given and the class instance
                has no observations stored.
        """
        if observation_data is None:
            if len(self.observations[0]) == 0:
                raise ValueError("User must either generate observation data using \
                                  collect_random_observations or provide data.")
            else:
                observation_data = self.observations
        
        self.summer_observations = []
        self.winter_observations = []

        if self.episodes is not None:
            for i in range(self.episodes):
                this_observation_block = observation_data[i]
                this_n = len(this_observation_block)

                this_first_season = this_observation_block[:this_n//4]
                this_summer = this_observation_block[this_n//4:3*this_n//4]
                this_last_season = this_observation_block[3*this_n//4:]
                this_winter = np.vstack((this_first_season, this_last_season))

                if i == 0:
                    self.summer_observations = this_summer
                    self.winter_observations = this_winter
                else:
                    self.summer_observations = np.vstack((self.summer_observations, this_summer))
                    self.winter_observations = np.vstack((self.winter_observations, this_winter))
        else:
            num_observations = observation_data.shape[0]
            self.summer_observations = observation_data[num_observations//4:3*num_observations//4, :]
            self.winter_observations = np.vstack((observation_data[:num_observations//4, :], observation_data[3*num_observations//4:, :]))

        # (len_of_season x num_episodes, dim_of_obs_vector)
        self.summer_observations = np.array(self.summer_observations)
        self.winter_observations = np.array(self.winter_observations)

        return self.summer_observations, self.winter_observations

    def get_nearest_block_size(self, obs_length: int, block_size: int):
        """
        Finds nearest block size to perfectly divide the length of the observation data.

        Args:
            obs_length: The number of observation vectors in the data.
            block_size: The desired block size.

        Returns:
            (one of) low, high: The nearest block size to perfectly divide the length
                of the data.
        """
        low = block_size
        high = block_size
        while obs_length%low != 0 and obs_length%high != 0:
            low -= 1
            high += 1
            if low <= 1:
                low = 2
        if obs_length%low == 0: return low
        elif obs_length%high == 0: return high

    def get_empirical_dist(self, 
                           season: Optional[str] = None, 
                           this_season_observations: Optional[np.array] = None, 
                           block_size_on_split: int = 100):
        """
        Fits a multivariate normal distribution to each ambient feature.

        Args:
            season: The desired season. Can be `None` if this_season_observations is
                not None. Otherwise, can be `summer` or `winter` if data has been generated
                and split into seasons through split_observations_into_seasons.
            this_season_observations: The observation data for this season. Can be
                `None` if user has generated and split data.
            block_size_on_split: Desired block size for each ambient feature vector.
                May be altered to perfectly divide the length of the observation data.

        Returns:
            empirical_dists: The empirical distributions for each of the ambient
                features.
        
        Raises:
            ValueError if neither season nor this_season_observations is specified OR
                if the season is given but it is not "summer" or "winter"
        """
        if season is None and this_season_observations is None:
            raise ValueError("Either season or this_season_observations must be specified.")
        if season is not None and this_season_observations is None:
            if season == "summer":
                this_season_observations = self.summer_observations
            elif season == "winter":
                this_season_observations = self.winter_observations
            else:
                raise ValueError("Season must be either summer or winter.")
        
        this_season_observations = np.array(this_season_observations)
        num_obs, num_features = this_season_observations.shape
        block_size = self.get_nearest_block_size(num_obs, block_size_on_split)
        self.block_size = block_size

        mu_vectors = []
        cov_matrices = []
        for i in range(num_features):
            this_col = this_season_observations[:, i]
            reshaped_obs = this_col.reshape(block_size, num_obs // block_size, order="F")
            this_mu_vector = np.mean(reshaped_obs, axis=1)
            this_cov_mat = np.cov(reshaped_obs)
            mu_vectors.append(this_mu_vector)
            cov_matrices.append(this_cov_mat)

        empirical_dists = []
        for i in range(len(mu_vectors)):
            this_dist = scipy.stats.multivariate_normal(mean=mu_vectors[i], cov=cov_matrices[i], allow_singular=True)
            empirical_dists.append(this_dist)
        
        if season == "summer":
            self.summer_dists = empirical_dists
        elif season == "winter":
            self.winter_dists = empirical_dists

        return empirical_dists

    def draw_samples_from_dist(
        self, 
        num_samples: int, 
        season: Optional[str] = None, 
        dists: Optional[scipy.stats.rv_continuous] = None, 
        block_size: Optional[int] = None
    ):
        """
        Draw vector samples from fitted multivariate Gaussian.

        Args:
            num_samples: The number of desired samples.
            season: The desired season. Can be `None`, `summer`, or `winter`.
            dists: The empirical distributions. Can be `None` if instance has
                generated empirical distributions through get_empirical_dist.

        Returns:
            samples: The samples generated from the fitted distributions.
        
        Raises:
            ValueError if season and dists is not given or if a season is given
                and is valid but there is no distribution for that season
        """
        if season is None and dists is None:
            raise ValueError("Either season or dist must be supplied.")
        if dists is None:
            if season == "summer":
                if self.summer_dists is None:
                    raise ValueError("No summer dist available; call get_empirical_dist")
                dists = self.summer_dists
            elif season == "winter":
                if self.winter_dists is None:
                    raise ValueError("No winter dist available; call get_empirical_dist")
                dists = self.winter_dists
            else:
                raise ValueError("Season must be either summer or winter")
        
        if block_size is None:
            block_size = self.block_size
        
        num_dists = len(dists)
        num_blocks = num_samples // block_size

        samples = []
        for i in range(num_dists):
            this_dist = dists[i]
            this_samples = this_dist.rvs(size=num_blocks)

            this_samples = this_samples.reshape(-1, 1)
            if i == 0:
                samples = this_samples
            else:
                samples = np.hstack((samples, this_samples))
        
        return samples