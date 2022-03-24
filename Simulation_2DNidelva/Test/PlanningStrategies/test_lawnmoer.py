import matplotlib.pyplot as plt

from GOOGLE.Simulation_2DSquare.PlanningStrategies.Lawnmower import LawnMowerPlanning
from GOOGLE.Simulation_2DSquare.Plotting.plotting_func import *
from GOOGLE.Simulation_2DSquare.GPKernel.GPKernel import *
from GOOGLE.Simulation_2DSquare.Tree.Location import *
from GOOGLE.Simulation_2DSquare.Config.Config import *
from GOOGLE.Simulation_2DSquare.Tree.Knowledge import Knowledge


foldername = PATH_REPLICATES + "R_{:03d}/lawnmower/".format(0)
checkfolder(foldername)


starting_location = Location(0, 0)
ending_location = Location(0, 1)
polygon_border = np.array(BORDER)
polygon_obstacles = np.array(OBSTACLES)
budget = BUDGET
# stepsizes = np.arange(200, 2500, 200)
stepsizes = [.2]

for stepsize in stepsizes:
    knowledge = Knowledge(starting_location=starting_location, ending_location=ending_location,
                          polygon_border=polygon_border, polygon_obstacles=polygon_obstacles, budget=budget,
                          step_size_lawnmower=stepsize)
    t = LawnMowerPlanning(knowledge=knowledge)
    t.get_lawnmower_path()
    dist = t.get_distance_of_trajectory()
    t.get_refined_trajectory(stepsize=DISTANCE_NEIGHBOUR)
    path = np.array(t.lawnmower_trajectory)
    plt.plot(path[:, 0], path[:, 1],'k.-')
    path = np.array(t.lawnmower_refined_trajectory)
    plt.plot(path[:, 0], path[:, 1], 'g.-')
    plt.plot(polygon_obstacles[0][:, 0], polygon_obstacles[0][:, 1], 'r-')
    plt.plot(polygon_border[:, 0], polygon_border[:, 1], 'r-')
    plt.title("Lawnmower with step size {:.2f}m".format(stepsize))
    plt.gcf().text(0.15, 0.8, "Distance: {:.1f}m".format(dist), fontsize=14)
    plt.savefig(foldername+"lawnmower_{:d}".format(int(stepsize * 100)))
    plt.show()
    plt.close("all")




