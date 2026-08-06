"""Random baseline agent — samples uniformly from the flattened action space."""


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()
