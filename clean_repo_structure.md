# Clean Repository Structure

This document outlines the recommended structure for the cleaned repository, organizing the code into clear sections for both simulation and field testing.

## Directory Structure

```
RRTCV/
├── README.md                  # Main documentation
├── requirements.txt           # Dependencies
├── .gitignore                 # Git ignore file
├── data/                      # Essential data files
│   ├── csv/                   # CSV data files
│   │   ├── polygon_border.csv # Operational area boundary
│   │   └── polygon_obstacle.csv # Obstacle definitions
│   └── simulation_results/    # Sample simulation results
├── src/                       # Source code
│   ├── common/                # Common code used by both simulation and field tests
│   │   ├── Config.py          # Configuration parameters
│   │   ├── Field.py           # Field representation
│   │   ├── WGS.py             # WGS coordinate conversions
│   │   ├── GRF/               # Gaussian Random Field implementation
│   │   │   ├── GRF.py         # GRF class
│   │   │   └── __init__.py
│   │   ├── CostValley/        # Cost Valley implementation
│   │   │   ├── CostValley.py  # Cost Valley class
│   │   │   ├── Budget.py      # Budget constraints
│   │   │   ├── Direction.py   # Direction constraints
│   │   │   └── __init__.py
│   │   └── Visualiser/        # Visualization tools
│   │       ├── ValleyPlotter.py # Cost Valley visualization
│   │       ├── TreePlotter.py # RRT* tree visualization
│   │       └── __init__.py
│   ├── planners/              # Path planning algorithms
│   │   ├── Planner.py         # Base planner class
│   │   ├── RRTSCV/            # RRT* with Cost Valley
│   │   │   ├── RRTStarCV.py   # RRT* with Cost Valley implementation
│   │   │   ├── TreeNode.py    # Tree node class
│   │   │   └── __init__.py
│   │   ├── Myopic2D/          # Myopic planner
│   │   │   ├── Myopic2D.py    # Myopic planner implementation
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── simulation/            # Simulation code
│   │   ├── run_simulation.py  # Main simulation runner
│   │   ├── Simulator.py       # Base simulator class
│   │   ├── Simulator_RRTStar.py # RRT* simulator
│   │   ├── Simulator_Myopic2D.py # Myopic simulator
│   │   ├── CTD.py             # CTD sensor simulation
│   │   └── __init__.py
│   └── field_test/            # Field test code
│       ├── run_field_test.py  # Main field test runner
│       ├── AUV.py             # AUV interface
│       ├── EDA.py             # Experiment data analysis
│       └── __init__.py
└── notebooks/                 # Jupyter notebooks
    ├── simulation_analysis.ipynb # Analysis of simulation results
    └── field_test_analysis.ipynb # Analysis of field test results
```

## Key Components

### Common Code

The `common` directory contains code shared between simulation and field tests:
- Configuration management
- Field representation
- Coordinate transformations
- Gaussian Random Field implementation
- Cost Valley implementation
- Visualization tools

### Planners

The `planners` directory contains path planning algorithms:
- RRT* with Cost Valley (RRTSCV)
- Myopic planner
- Base planner class

### Simulation

The `simulation` directory contains code for running simulations:
- Simulator classes for different planners
- Sensor simulation
- Main simulation runner

### Field Test

The `field_test` directory contains code for running field tests:
- AUV interface
- Experiment data analysis
- Main field test runner

## How to Run

### Simulation

To run a simulation:

```bash
cd src/simulation
python run_simulation.py
```

### Field Test

To run a field test:

```bash
cd src/field_test
python run_field_test.py
```

## Data Files

Essential data files are stored in the `data` directory:
- CSV files defining operational areas and obstacles
- Sample simulation results for reference 