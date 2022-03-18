

from GOOGLE.Simulation_2DNidelva.PlanningStrategies.RRTStar import RRTStar
from GOOGLE.Simulation_2DNidelva.Tree.Knowledge import Knowledge
from GOOGLE.Simulation_2DNidelva.Tree.Location import Location
from usr_func import *


PATH_OPERATION_AREA = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Simulation_2DNidelva/Config/Polygon_border.csv"
PATH_MUNKHOLMEN = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Simulation_2DNidelva/Config/Polygon_obstacle.csv"
DISTANCE_STEPSIZE = 500
DISTANCE_TOLERANCE = 500
DISTANCE_NEIGHBOUR = 600
GOAL_SAMPLE_RATE = .01
MAX_ITER = 400
BUDGET = 6000

polygon_border = pd.read_csv(PATH_OPERATION_AREA).to_numpy()
polygon_obstacle = pd.read_csv(PATH_MUNKHOLMEN).to_numpy()

starting_location = Location(63.440887, 10.354804)
ending_location = Location(63.455674, 10.429927)

knowledge = Knowledge(starting_location=starting_location, ending_location=ending_location, goal_location=ending_location,
                      goal_sample_rate=GOAL_SAMPLE_RATE, polygon_border=polygon_border, polygon_obstacle=polygon_obstacle,
                      step_size=DISTANCE_STEPSIZE, maximum_iteration=MAX_ITER, distance_neighbour_radar=DISTANCE_NEIGHBOUR,
                      distance_tolerance=DISTANCE_TOLERANCE, budget=BUDGET, kernel=None)

rrtstar = RRTStar(knowledge)
rrtstar.expand_trees()
rrtstar.trajectory = []
rrtstar.get_shortest_trajectory()
rrtstar.plot_tree()
plt.show()

#%%
loc1 = Location(63.45, 10.375)
loc2 = Location(63.46, 10.4)

n1 = 50
n2 = 250
polygon_new = rrtstar.knowledge.polygon_border[n1:n2]
plt.plot(polygon_border[n1:n2, 1], polygon_border[n1:n2, 0], 'k.-')
plt.plot([loc1.lon, loc2.lon], [loc1.lat, loc2.lat], 'r-')
plt.show()
line = LineString([(loc1.lat, loc1.lon),
                   (loc2.lat, loc2.lon)])

print(rrtstar.knowledge.polygon_border_path.intersects(line))
path = Polygon(polygon_new)








