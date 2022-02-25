

from GOOGLE.RRTStar.RRTStar import *
from usr_func import *


PATH_OPERATION_AREA = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Config/OpArea.csv"
PATH_MUNKHOLMEN = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Config/Munkholmen.csv"
DISTANCE_LATERAL = 150
DISTANCE_VERTICAL = .5
DISTANCE_TOLERANCE = 50
DISTANCE_NEIGHBOUR = 200
DEPTH = [0, 2, 4]
GOAL_SAMPLE_RATE = .05
MAXNUM = 1000

polygon = pd.read_csv(PATH_OPERATION_AREA).to_numpy()
munkholmen = pd.read_csv(PATH_MUNKHOLMEN).to_numpy()


location_TBS = Location(63.440752, 10.349210, 0)
location_LADE = Location(63.457086, 10.440334, 0)


rrtConfig = RRTConfig(polygon_within=polygon, polygon_without=munkholmen, depth=DEPTH, starting_location=location_TBS,
                      ending_location=location_LADE, goal_sample_rate=GOAL_SAMPLE_RATE, step_lateral=DISTANCE_LATERAL,
                      step_vertical=DISTANCE_VERTICAL, maximum_num=MAXNUM, neighbour_radius=DISTANCE_NEIGHBOUR,
                      distance_tolerance=DISTANCE_TOLERANCE)

rrtstar = RRTStar(rrtConfig)
rrtstar.plot_tree()



