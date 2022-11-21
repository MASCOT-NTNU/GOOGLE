

class SimulationResultContainer:

    def __init__(self, strategyname):
        self.strategyname = strategyname
        self.expected_integrated_bernoulli_variance = []
        self.root_mean_squared_error = []
        self.expected_variance = []
        self.distance_travelled = []
        self.continuous_ranked_probability_score = []

    def append(self, knowledge):
        print(knowledge.integrated_bernoulli_variance)
        print(knowledge.distance_travelled)
        print(knowledge.rmse)
        self.expected_integrated_bernoulli_variance.append(knowledge.integrated_bernoulli_variance)
        self.root_mean_squared_error.append(knowledge.rmse)
        self.expected_variance.append(knowledge.uncertainty)
        self.distance_travelled.append(knowledge.distance_travelled)
        self.continuous_ranked_probability_score.append(knowledge.crps)

