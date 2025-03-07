# RRTCV
# RRT*-Enhanced Long-Horizon Path Planning for AUV Adaptive Sampling using a Cost Valley

## Overview
This repository contains the code and data associated with the research paper:

**RRT*-Enhanced Long-Horizon Path Planning for AUV Adaptive Sampling using a Cost Valley**  
Authors: Yaolin Ge, Jo Eidsvik, André Julius Hovd Olaisen  
Affiliation: Department of Mathematical Sciences, Norwegian University of Science and Technology (NTNU), Trondheim, Norway  

This study presents an adaptive sampling method using autonomous underwater vehicles (AUVs) for long-horizon path planning. It introduces a flexible cost valley concept combined with a non-myopic RRT* planner, optimizing oceanographic sampling while ensuring real-time computations on AUVs.

## Key Features
- **Adaptive Sampling with Multiple Objectives**: Combines variance reduction and classification error minimization.
- **Non-Myopic Path Planning**: Uses an RRT* strategy instead of traditional greedy (myopic) methods.
- **Cost Valley Concept**: Integrates operational constraints and informative sampling criteria into a unified weighted cost function.
- **Field Deployment**: Successfully tested in a 2.5-hour AUV mission in the Trondheim fjord, Norway.

## Repository Contents
- `code/`: Implementation of the RRT*-based long-horizon planner and cost valley calculations.
- `data/`: Sample datasets used for simulation and field trials.
- `notebooks/`: Jupyter notebooks for analyzing simulation results and visualizing cost fields.
- `docs/`: Additional documentation and references.

## Installation
Clone the repository:
```sh
git clone https://github.com/MASCOT-NTNU/GOOGLE.git
cd GOOGLE
```

Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
To run a simulation with predefined parameters:
```sh
python simulate.py --config config/default.yaml
```

To visualize results:
```sh
python visualize.py --input data/simulation_results.json
```

## Citation
If you use this work in your research, please cite:
```
@article{Ge2025,
  author = {Yaolin Ge, Jo Eidsvik, André Julius Hovd Olaisen},
  title = {RRT*-Enhanced Long-Horizon Path Planning for AUV Adaptive Sampling using a Cost Valley},
  journal = {Knowledge-Based Systems},
  year = {2025},
  doi = {TBD}
}
```

## Contact
For questions or collaborations, please contact:
- Yaolin Ge: geyaolin@gmail.com
- Jo Eidsvik: jo.eidsvik@ntnu.no
- André Julius Hovd Olaisen: andre.j.h.olaisen@ntnu.no

---

This work was supported by the Norwegian Research Council (RCN) through the MASCOT project (305445). We thank NTNU AURLab and SINTEF Ocean for their support and data access.

<!-- # GOOGLE
GOOd GLobal Extented-horizon path planning

# EDA analysis reminders
There are three important parameters used to adjust SINMOD data and for EIBV exploration.
- $\beta_0 = 0.26095833$
- $\beta_1 = 0.99898364$
- $threshold = 26.81189868$

Here comes the parameters used in the actual test for GRF kernel.
- $\sigma = 1.5$
- $\tau = .4$
- `lateral range = 700m`
- $\phi = 4.5/700$

# HITL test:
Open 4 iterfaces either through tmux or `Ctrl+Alt+T`.
---
- `cd ~/catkin_ws/`
- `source devel/setup.bash`
---
- `cd ~/dune_all/build`
- `./dune -c lauv-simulator-1 -p Simulation`
---
- `cd ~/catkin_ws/`
- `source devel/setup.bash`
- `roslaunch src/imc_ros_interface/launch/bridge.launch `
---
- `cd Simulation_2DNidelva/HITL/`
- `python3 GOOGLELauncher.py` -->
