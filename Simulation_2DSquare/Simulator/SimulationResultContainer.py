

class SimulationResultContainer:

    def __init__(self, strategyname):
        self.strategyname = strategyname
        self.expectedIntegratedBernoulliVariance = []
        self.rootMeanSquaredError = []
        self.expectedVariance = []
        self.distanceTravelled = []

    def append(self, knowledge):
        self.expectedIntegratedBernoulliVariance.append(knowledge.intergrated_bernoulli_variance)
        self.rootMeanSquaredError.append(knowledge.root_mean_squared_error)
        self.expectedVariance.append(knowledge.expected_variance)
        self.distanceTravelled.append(knowledge.distance_travelled)

