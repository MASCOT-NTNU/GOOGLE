#!/bin/bash

# Create the new directory structure
mkdir -p RRTCV/src/common/GRF
mkdir -p RRTCV/src/common/CostValley
mkdir -p RRTCV/src/common/Visualiser
mkdir -p RRTCV/src/planners/RRTSCV
mkdir -p RRTCV/src/planners/Myopic2D
mkdir -p RRTCV/src/simulation
mkdir -p RRTCV/src/field_test
mkdir -p RRTCV/data/csv
mkdir -p RRTCV/data/simulation_results
mkdir -p RRTCV/notebooks

# Copy README and requirements
cp README.md RRTCV/
cp requirements.txt RRTCV/
cp .gitignore RRTCV/

# Copy common files
cp Publication/src/Config.py RRTCV/src/common/
cp Publication/src/Field.py RRTCV/src/common/
cp Publication/src/WGS.py RRTCV/src/common/

# Copy GRF files
cp Publication/src/GRF/* RRTCV/src/common/GRF/

# Copy CostValley files
cp Publication/src/CostValley/CostValley.py RRTCV/src/common/CostValley/
cp Publication/src/CostValley/Budget.py RRTCV/src/common/CostValley/
cp Publication/src/CostValley/Direction.py RRTCV/src/common/CostValley/
touch RRTCV/src/common/CostValley/__init__.py

# Copy Visualiser files
cp Publication/src/Visualiser/ValleyPlotter.py RRTCV/src/common/Visualiser/
cp Publication/src/Visualiser/TreePlotter.py RRTCV/src/common/Visualiser/
touch RRTCV/src/common/Visualiser/__init__.py

# Copy planner files
cp Publication/src/Planner/Planner.py RRTCV/src/planners/
cp Publication/src/Planner/RRTSCV/RRTStarCV.py RRTCV/src/planners/RRTSCV/
cp Publication/src/Planner/RRTSCV/TreeNode.py RRTCV/src/planners/RRTSCV/
touch RRTCV/src/planners/RRTSCV/__init__.py
cp Publication/src/Planner/Myopic2D/* RRTCV/src/planners/Myopic2D/
touch RRTCV/src/planners/Myopic2D/__init__.py
touch RRTCV/src/planners/__init__.py

# Copy simulation files
cp Publication/src/Simulators/Simulator.py RRTCV/src/simulation/
cp Publication/src/Simulators/Simulator_RRTStar.py RRTCV/src/simulation/
cp Publication/src/Simulators/Simulator_Myopic2D.py RRTCV/src/simulation/
cp Publication/src/Simulators/CTD.py RRTCV/src/simulation/
cp Publication/src/run_replicates.py RRTCV/src/simulation/run_simulation.py
touch RRTCV/src/simulation/__init__.py

# Copy field test files
cp Publication/src/Experiment/AUV.py RRTCV/src/field_test/
cp Publication/src/Experiment/EDA.py RRTCV/src/field_test/
touch RRTCV/src/field_test/run_field_test.py
touch RRTCV/src/field_test/__init__.py

# Copy data files
cp Publication/src/csv/polygon_border.csv RRTCV/data/csv/
cp Publication/src/csv/polygon_obstacle.csv RRTCV/data/csv/

# Copy notebooks
cp Publication/prior.ipynb RRTCV/notebooks/simulation_analysis.ipynb
cp Publication/truth.ipynb RRTCV/notebooks/field_test_analysis.ipynb

# Create __init__.py files
touch RRTCV/src/common/__init__.py
touch RRTCV/src/common/GRF/__init__.py

echo "Repository reorganization completed!" 