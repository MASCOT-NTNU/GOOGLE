

class SimulationResultContainer:

    def __init__(self, strategyname):
        self.strategyname = strategyname
        self.expected_integrated_bernoulli_variance = []
        self.root_mean_squared_error = []
        self.expected_variance = []
        self.distance_travelled = []

    def append(self, knowledge):
        print(knowledge.integrated_bernoulli_variance)
        print(knowledge.distance_travelled)
        print(knowledge.root_mean_squared_error)
        self.expected_integrated_bernoulli_variance.append(knowledge.integrated_bernoulli_variance)
        self.root_mean_squared_error.append(knowledge.root_mean_squared_error)
        self.expected_variance.append(knowledge.expected_variance)
        self.distance_travelled.append(knowledge.distance_travelled)

