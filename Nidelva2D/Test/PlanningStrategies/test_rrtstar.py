import os

from GOOGLE.Simulation_2DNidelva.PlanningStrategies.RRTStar import RRTStar
from GOOGLE.Simulation_2DNidelva.Tree.Knowledge import Knowledge
from GOOGLE.Simulation_2DNidelva.Tree.Location import *
from usr_func import *


polygon_border = os.getcwd() + "/GOOGLE/Simulation_2DNidelva/Config/polygon_border.csv"
polygon_obstacle = os.getcwd() + "/GOOGLE/Simulation_2DNidelva/Config/polygon_obstacle.csv"
DISTANCE_STEPSIZE = 250
DISTANCE_TOLERANCE = 50
DISTANCE_NEIGHBOUR = 50
GOAL_SAMPLE_RATE = .01
MAX_ITER = 400
BUDGET = 6000

polygon_border_xy = pd.read_csv(polygon_border)[['x', 'y']].to_numpy()
polygon_obstacle_xy = pd.read_csv(polygon_obstacle)[['x', 'y']].to_numpy()

starting_location = WGS2XY(LocationWGS(63.440887, 10.354804))
ending_location = WGS2XY(LocationWGS(63.455674, 10.429927))

knowledge = Knowledge(starting_location=starting_location, ending_location=ending_location,
                      goal_location=ending_location, goal_sample_rate=GOAL_SAMPLE_RATE,
                      polygon_border_xy=polygon_border_xy, polygon_obstacle_xy=polygon_obstacle_xy,
                      step_size=DISTANCE_STEPSIZE, maximum_iteration=MAX_ITER,
                      distance_neighbour_radar=DISTANCE_NEIGHBOUR,
                      distance_tolerance=DISTANCE_TOLERANCE, budget=BUDGET)

rrtstar = RRTStar(knowledge)
rrtstar.expand_trees()
rrtstar.trajectory_plot = []
rrtstar.get_shortest_trajectory()
rrtstar.plot_tree()
plt.show()

#%%
# loc1 = LocationWGS(63.45, 10.375)
# loc2 = LocationWGS(63.46, 10.4)
#
# n1 = 50
# n2 = 250
# polygon_new = rrtstar.knowledge.polygon_border_xy[n1:n2]
# plt.plot(polygon_border_xy[n1:n2, 1], polygon_border_xy[n1:n2, 0], 'k.-')
# plt.plot([loc1.lon, loc2.lon], [loc1.lat, loc2.lat], 'r-')
# plt.show()
# line = LineString([(loc1.lat, loc1.lon),
#                    (loc2.lat, loc2.lon)])
#
# print(rrtstar.knowledge.polygon_border_shapely.intersects(line))
# path = Polygon(polygon_new)








