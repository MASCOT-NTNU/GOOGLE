"""
This script generates the potential random locations within a certain constraint
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-23
"""


from GOOGLE.Simulation_2DNidelva.Tree.Location import *
from GOOGLE.Simulation_2DNidelva.Config.Config import *
from usr_func import *

filepath_polygon_border = FILEPATH + "/Config/Polygon_border.csv"
filepath_polygon_obstacle = FILEPATH + "/Config/Polygon_obstacle.csv"

polygon_border = pd.read_csv(filepath_polygon_border)[['x', 'y']].to_numpy()
polygon_obstacle = pd.read_csv(filepath_polygon_obstacle)[['x', 'y']].to_numpy()
polygon_border_shapely = Polygon(polygon_border)
polygon_obstacle_shapely = Polygon(polygon_obstacle)

x_min, y_min = map(np.amin, [polygon_border[:, 0], polygon_border[:, 1]])
x_max, y_max = map(np.amax, [polygon_border[:, 0], polygon_border[:, 1]])

NUM = int(1e6)
x = np.random.uniform(x_min, x_max, NUM)
y = np.random.uniform(y_min, y_max, NUM)

locs = np.hstack((vectorise(x), vectorise(y)))

ind_within_border = [polygon_border_shapely.contains(Point(location[0], location[1])) for location in locs]
locs = locs[ind_within_border]
ind_not_collided = [not polygon_obstacle_shapely.contains(Point(location[0], location[1])) for location in locs]
locs = locs[ind_not_collided]
print("remaining samples: ", locs.shape)
print("Acceptance rate: {:.1f}%".format(100 * locs.shape[0] / NUM))

plt.axhline(x_min)
plt.axhline(x_max)
plt.axvline(y_min)
plt.axvline(y_max)
plt.plot(locs[:, 1], locs[:, 0], 'k.', alpha=.1)
plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'k-', linewidth=1)
plt.plot(polygon_obstacle[:, 1], polygon_obstacle[:, 0], 'k-', linewidth=1)
plt.show()

#%% save random locations
np.save(FILEPATH+"Config/RandomLocations.npy", locs)

#%% test saved locations
t = np.load(FILEPATH+"Config/RandomLocations.npy")



