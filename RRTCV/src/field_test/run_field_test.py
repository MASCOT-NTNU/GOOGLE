#!/usr/bin/env python3
"""
Field Test Runner for RRT*-CV

This script runs the field test for the RRT*-CV algorithm.
It interfaces with the AUV and performs adaptive sampling in the field.

Author: Yaolin Ge
Email: geyaolin@gmail.com
"""

import sys
import os
# Add the parent directory to the path so we can import from common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from field_test.AUV import AUV
from field_test.EDA import EDA
from common.Config import Config
from common.GRF.GRF import GRF
from planners.RRTSCV.RRTStarCV import RRTStarCV
from common.CostValley.CostValley import CostValley
from common.CostValley.Budget import Budget
import numpy as np
import time


def main():
    """
    Main function to run the field test.
    """
    print("Starting RRT*-CV field test...")
    
    # Initialize configuration
    config = Config()
    
    # Initialize AUV
    auv = AUV()
    
    # Initialize GRF
    grf = GRF(sigma=1.5, nugget=0.4, approximate_eibv=False, fast_eibv=True)
    
    # Initialize Cost Valley
    cost_valley = CostValley()
    
    # Initialize Budget
    budget = Budget()
    
    # Initialize RRT*-CV planner
    planner = RRTStarCV(
        start=config.get_loc_start(),
        goal=config.get_loc_end(),
        cost_valley=cost_valley,
        budget=budget,
        polygon_border=config.get_polygon_border(),
        polygon_obstacle=config.get_polygon_obstacle(),
        polygon_border_shapely=config.get_polygon_border_shapely(),
        polygon_obstacle_shapely=config.get_polygon_obstacle_shapely(),
        line_border_shapely=config.get_line_border_shapely(),
        line_obstacle_shapely=config.get_line_obstacle_shapely(),
        waypoint_distance=config.get_waypoint_distance(),
        neighbour_distance=120,
        max_connection_distance=240,
        max_nodes=1000,
        debug=True
    )
    
    # Initialize EDA for data analysis
    eda = EDA()
    
    # Run the field test
    print("Running field test...")
    
    # Set up the mission parameters
    num_steps = config.get_num_steps()
    
    # Main mission loop
    for i in range(num_steps):
        print(f"Step {i+1}/{num_steps}")
        
        # Get current AUV position
        current_position = auv.get_position()
        
        # Plan path using RRT*-CV
        path = planner.plan()
        
        # Send path to AUV
        auv.set_path(path)
        
        # Wait for AUV to reach the next waypoint
        while not auv.reached_waypoint():
            time.sleep(1)
        
        # Collect data at current position
        data = auv.collect_data()
        
        # Update GRF with new data
        grf.update(data)
        
        # Update Cost Valley with new GRF
        cost_valley.update(grf)
        
        # Analyze data
        eda.analyze_step(i, current_position, data, path)
    
    # Mission complete
    print("Field test completed successfully!")
    
    # Save results
    eda.save_results()


if __name__ == "__main__":
    main() 