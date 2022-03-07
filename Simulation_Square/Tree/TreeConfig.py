

class RRTConfig:

    def __init__(self, starting_location=None, ending_location=None,
                 goal_sample_rate=None, step=None, GPKernel=None,
                 budget=None, goal_location=None):
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.goal_sample_rate = goal_sample_rate
        self.step = step
        self.GPKernel = GPKernel
        self.budget = budget
        self.goal_location = goal_location
        pass
