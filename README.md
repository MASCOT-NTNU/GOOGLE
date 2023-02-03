# GOOGLE
GOOd GLobal Extented-horizon path planning

# EDA analysis reminders
There are three important parameters used to adjust SINMOD data and for EIBV exploration.
- $\beta_0 = 0.26095833$
- $\beta_1 = 0.99898364$
- $threshold = 26.81189868$

Here comes the parameters used in the actual test for GRF kernel.
- `sigma = 1.5`
- `nugget = .4`
- `lateral range = 700m`



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
- `python3 GOOGLELauncher.py`
