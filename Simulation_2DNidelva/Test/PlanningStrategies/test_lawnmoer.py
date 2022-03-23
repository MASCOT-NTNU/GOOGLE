import matplotlib.pyplot as plt

from GOOGLE.Simulation_2DNidelva.PlanningStrategies.Lawnmower import LawnMowerPlanning
from GOOGLE.Simulation_2DNidelva.Plotting.plotting_func import *
from GOOGLE.Simulation_2DNidelva.GPKernel.GPKernel import *
from GOOGLE.Simulation_2DNidelva.Tree.Location import *
from GOOGLE.Simulation_2DNidelva.Config.Config import *
from GOOGLE.Simulation_2DNidelva.Tree.Knowledge import Knowledge


foldername = PATH_REPLICATES + "R_{:03d}/lawnmower/".format(0)
checkfolder(foldername)


starting_location = Location(63.440887, 10.354804)
ending_location = Location(63.455674, 10.429927)
polygon_border = pd.read_csv(PATH_BORDER).to_numpy()
polygon_obstacle = pd.read_csv(PATH_OBSTACLE).to_numpy()
budget = BUDGET
# stepsizes = np.arange(200, 2500, 200)
stepsizes = [1400]

for stepsize in stepsizes:
    knowledge = Knowledge(starting_location=starting_location, ending_location=ending_location,
                          polygon_border=polygon_border, polygon_obstacle=polygon_obstacle, budget=budget,
                          step_size_lawnmower=stepsize)
    t = LawnMowerPlanning(knowledge=knowledge)
    t.get_lawnmower_path()
    dist = t.get_distance_of_trajectory()

    path = np.array(t.lawnmower_trajectory)
    plt.plot(path[:, 1], path[:, 0],'k.-')
    plt.plot(polygon_obstacle[:, 1], polygon_obstacle[:, 0], 'r-')
    plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'r-')
    plt.title("Lawnmower with step size {:d}m".format(stepsize))
    plt.gcf().text(0.15, 0.8, "Distance: {:.1f}m".format(dist), fontsize=14)
    plt.savefig(foldername+"lawnmower_{:d}".format(stepsize))
    plt.show()
    plt.close("all")




