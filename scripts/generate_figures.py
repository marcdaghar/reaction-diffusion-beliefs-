#!/usr/bin/env python3
"""
Generate all figures from saved simulation results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from src.visualization import FigureGenerator

def main():
    """
    Load saved results and generate all figures.
    """
    print("Loading saved results...")
    
    # Load 2D results
    with open('data/results_2d.pkl', 'rb') as f:
        results_2d = pickle.load(f)
    
    # Load velocity results
    with open('data/velocity_results.pkl', 'rb') as f:
        velocity_results = pickle.load(f)
    
    # Load phase results
    with open('data/phase_results.pkl', 'rb') as f:
        phase_results = pickle.load(f)
    
    # Load network results
    with open('data/network_results.pkl', 'rb') as f:
        network_results = pickle.load(f)
    
    # Generate all figures
    print("Generating all figures...")
    fig_gen = FigureGenerator()
    fig_gen.figure_all(results_2d, velocity_results, phase_results, network_results)
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()
