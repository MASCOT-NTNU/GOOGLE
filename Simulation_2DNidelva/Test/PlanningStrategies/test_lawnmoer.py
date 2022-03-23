from GOOGLE.Simulation_2DNidelva.PlanningStrategies.Lawnmower import LawnMowerPlanning
from GOOGLE.Simulation_2DNidelva.Plotting.plotting_func import *
from GOOGLE.Simulation_2DNidelva.GPKernel.GPKernel import *
from GOOGLE.Simulation_2DNidelva.Tree.Location import *
from GOOGLE.Simulation_2DNidelva.Config.Config import *
from GOOGLE.Simulation_2DNidelva.Tree.Knowledge import Knowledge


# foldername = PATH_REPLICATES + "R_{:03d}/rrtstar/".format(0)
# checkfolder(foldername)


starting_location = Location(63.455674, 10.429927)
ending_location = Location(63.440887, 10.354804)
polygon_border = pd.read_csv(PATH_BORDER).to_numpy()
polygon_obstacle = pd.read_csv(PATH_OBSTACLE).to_numpy()
budget = BUDGET

t = LawnMowerPlanning(starting_location=starting_location, ending_location=ending_location,
                      polygon_border=polygon_border, polygon_obstacle=polygon_obstacle, budget=budget,
                      stepsize=500, width=600)

t.get_lawnmower_path()

path = np.array(t.lawn_mower_path_2d)
plt.plot(path[:, 1], path[:, 0],'k.-')
plt.show()




