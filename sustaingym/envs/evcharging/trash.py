                # If action_type == 'discrete', entries should be in {0, 1, 2, 3, 4}.
                # If action_type == 'continuous', entries should be in range [0, 4].
                # Multiply action by ACTION_SCALE_FACTOR to convert to Amps.

        # # Define observation space and action space
        # self.observation_range = {
        #     'est_departures':  (-self.max_timestep, self.max_timestep),  # negative if estimated is passed
        #     'demands':         (0, data_generator.requested_energy_cap),
        #     'prev_moer':       (0, 1),
        #     'forecasted_moer': (0, 1),
        #     'timestep':        (0, self.max_timestep),
        # }


    # def raw_observation(self, obs: dict[str, np.ndarray], key: str) -> np.ndarray:
    #     """Returns unnormalized form of observation.

    #     During the step() function, observations are normalized between 0 and
    #     1 using ``self.observation_range``. This function "undoes" the
    #     normalization.

    #     Args:
    #         obs: state, see step()
    #         key: attribute of obs to get

    #     Returns:
    #         unnormalized observation
    #     """
    #     return obs[key] * (self.observation_range[key][1] - self.observation_range[key][0]) + self.observation_range[key][0]



        # for s in self.obs:  # modify self.obs in place
        #     self.obs[s] -= self.observation_range[s][0]
        #     self.obs[s] /= self.observation_range[s][1] - self.observation_range[s][0]