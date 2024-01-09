import numpy as np
import pandas as pd
import scipy

class StochasticUncontrollableGenerator():
    def __init__(self, 
                 building_env = None, 
                 num_episodes: int = None):
        """
        Initializes a generator class for the uncontrollable features in BuildingEnv.

        :param building_env: The instance of the BuildingEnv environment.
        :param num_episodes: The number of episodes over which to collect observations
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

    def fit_and_generate_observations(self, 
                                      num_rows: int, 
                                      season: str,
                                      weather_data: pd.DataFrame, 
                                      block_size_on_split: int = 100):
        """
        """
        num_observations = len(weather_data)
        winter_observations = pd.concat([weather_data.iloc[:num_observations//4, :],
                                         weather_data.iloc[3*num_observations//4:, :]],
                                         axis=0).to_numpy()
        summer_observations = weather_data.iloc[num_observations//4:3*num_observations//4, :].to_numpy()

        summer_dist = self.get_empirical_dist("summer", summer_observations, block_size_on_split)
        winter_dist = self.get_empirical_dist("winter", winter_observations, block_size_on_split)

        if season == "summer":
            samples = self.draw_samples_from_dist(num_rows, "summer")
            return samples
        elif season == "winter":
            samples = self.draw_samples_from_dist(num_rows, "winter")
            return samples
        else:
            raise ValueError("Season must be either `summer` or `winter`")
    
    def collect_data_and_fit(self, 
                             season: str, 
                             num_rows: int,
                             block_size_on_split: int = 100):
        """
        Collects random observations and fits distribution to it.

        :param block_size_on_split: Desired size of blocks for each time period over which
            samples will be generated. May be resized to perfectly divide length of
            observations.
        """
        assert self.episodes is not None

        self.collect_random_observations(self.env, self.episodes)
        self.split_observations_into_seasons(self.observations)
        self.summer_dists = self.get_empirical_dist("summer", 
                                                    self.summer_observations, 
                                                    block_size_on_split=block_size_on_split)
        self.winter_dists = self.get_empirical_dist("winter", 
                                                    self.winter_observations,
                                                    block_size_on_split=block_size_on_split)

    def get_observations(self):
        """
        Returns collected observations.
        """
        if len(self.observations[0]) == 0:
            print("Observations are empty. Call collect_random_observations to \
                  generate them.")
        return self.observations

    def collect_random_observations(self, building_env, num_episodes: int = 1):
        """
        Collects random observations from given BuildingEnvironment.

        :param building_env: Instance of the BuildingEnv.
        :param num_episodes: Number of episodes over which to collect observations.

        :return observations: numpy array of the observations.
        """
        assert num_episodes > 0

        self.observations = [[]]

        for i in range(num_episodes):
            terminated = False
            obs, _ = building_env.reset()
            while not terminated:
                action = building_env.action_space.sample()
                obs, _, terminated, _, _ = building_env.step(action)
                self.observations[i].append(obs)
            self.observations.append([])
        
        self.observations = self.observations[:-1]
        self.observations = np.array(self.observations)

        # only these 3 features are entirely functions of the environment
        self.observations = self.observations[:, :, -4:-1]

        return self.observations

    def split_observations_into_seasons(self, observation_data: np.array = None):
        """
        Splits observation data into summer and winter seasons.

        :param observation_data: The collected observation data. Can be `None` if 
            collect_random_observations has been called.
        
        :return summer_observations: Summer ambient features.
        :return winter_observations: Winter ambient features.
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

    def get_nearest_block_size(self, obs_length, block_size):
        """
        Finds nearest block size to perfectly divide the length of the observation data.

        :param obs_length: The number of observation vectors in the data.
        :param block_size: The desired block size.

        :return (one of) low, high: The nearest block size to perfectly divide the length
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
                           season: str = None, 
                           this_season_observations: np.array = None, 
                           block_size_on_split: int = 100):
        """
        Fits a multivariate normal distribution to each ambient feature.

        :param season: The desired season. Can be `None` if this_season_observations is
            not None. Otherwise, can be `summer` or `winter` if data has been generated
            and split into seasons through split_observations_into_seasons.
        :param this_season_observations: The observation data for this season. Can be
            `None` if user has generated and split data.
        :param block_size_on_split: Desired block size for each ambient feature vector.
            May be altered to perfectly divide the length of the observation data.

        :return empirical_dists: The empirical distributions for each of the ambient
            features.
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

    def draw_samples_from_dist(self, num_samples, season=None, dists=None, block_size=None):
        """
        Draw vector samples from fitted multivariate Gaussian.

        :param num_samples: The number of desired samples.
        :param season: The desired season. Can be `None`, `summer`, or `winter`.
        :param dists: The empirical distributions. Can be `None` if instance has
            generated empirical distributions through get_empirical_dist.

        :return samples: The samples generated from the fitted distributions.
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

            this_size = this_samples.shape[0] * this_samples.shape[1]

            this_samples = this_samples.reshape(this_size, 1, order="F")
            if i == 0:
                samples = this_samples
            else:
                samples = np.hstack((samples, this_samples))
        
        return samples