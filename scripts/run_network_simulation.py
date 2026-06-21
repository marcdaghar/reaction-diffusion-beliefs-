#!/usr/bin/env python3
"""
Run network ABM simulations for different topologies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from src.abm_network import ABMNetwork
from src.visualization import FigureGenerator

def run_network_simulations():
    """
    Run ABM on different network topologies.
    """
    print("=" * 60)
    print("Running Network ABM Simulations...")
    print("=" * 60)
    
    n_agents = 1000
    results = {}
    
    # Erdős-Rényi
    print("\n  Running Erdős-Rényi network...")
    model_er = ABMNetwork.create_er_network(n_agents, p=0.01)
    alpha0 = model_er.initialize_cluster(cluster_size=20)
    results['ER'] = model_er.run_simulation(alpha0, dt=0.01, T=500, save_snapshots=False)
    
    # Watts-Strogatz
    print("\n  Running Watts-Strogatz network...")
    model_ws = ABMNetwork.create_ws_network(n_agents, k=10, p=0.1)
    alpha0 = model_ws.initialize_cluster(cluster_size=20)
    results['WS'] = model_ws.run_simulation(alpha0, dt=0.01, T=500, save_snapshots=False)
    
    # Barabási-Albert
    print("\n  Running Barabási-Albert network...")
    model_ba = ABMNetwork.create_ba_network(n_agents, m=5)
    alpha0 = model_ba.initialize_cluster(cluster_size=20)
    results['BA'] = model_ba.run_simulation(alpha0, dt=0.01, T=500, save_snapshots=False)
    
    print("\nNetwork simulations complete!")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/network_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def main():
    """
    Main execution function.
    """
    # Run network simulations
    network_results = run_network_simulations()
    
    # Generate network figure
    print("\nGenerating network figure...")
    fig_gen = FigureGenerator()
    # We need to package the results for the figure generator
    fig_gen.figure_network_topology(network_results)
    
    print("Done!")

if __name__ == "__main__":
    main()
