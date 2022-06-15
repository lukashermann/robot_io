class BaseRecorder:
    def step(self, obs, action, next_obs, rew, done, info, record_info):
        """
        State transitions are (s, a, s', r).
        Depending on the use-case, the recorders may save only s or s'.

        Args:
            obs: Environment observation s.
            action: Action a chosen according to obs.
            next_obs: Environment observation s' after applying action a.
            rew: Environment reward.
            done: True if environment done.
            info: Environment info.
            record_info (dict): Info by input device (e.g. when to start and stop recording).
        """
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError
