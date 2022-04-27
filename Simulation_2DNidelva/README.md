# Procedure of running Simulation for Nidelva case

---
## Step I: generate polygons (obstacles, border)
- run `python3 PreConfig/SINMOD_Data_Region.py` to get SINMOD shape file.
- run `python3 PreConfig/OperationalArea.py` to get polygon_border.csv & polygon_obstacle.csv
- run `python3 PreConfig/gridGenerator.py` to get GRF grid.
- run `python3 dataInterpolator.py` to get SINMOD prior data
- run `python3 RandomLocationGenerator.py` to get pre-generated random locations for the tree.
