#!/usr/bin/env python3
"""
Run 2D grid ABM simulation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from src.abm_2d import ABM2D
from src.visualization import FigureGenerator

def run_2d_simulation():
    """
    Run ABM on 2D grid.
    """
    print("=" * 60)
    print("Running 2D Grid ABM Simulation...")
    print("=" * 60)
    
    # Initialize model
    model = ABM2D(nx=100, ny=100, D=0.1, beta=0.05, a=1.0, sigma=0.02)
    
    # Initialize with resistant cluster
    alpha0 = model.initialize_cluster(cluster_radius=8)
    
    # Run simulation
    results = model.run_simulation(alpha0, dt=0.01, T=500)
    
    print(f"\nSimulation complete!")
    print(f"  Final mean alpha: {np.mean(results['alpha_final']):.3f}")
    print(f"  Front position: {results['front_positions'][-1]:.1f}")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/results_2d.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def run_velocity_sweep():
    """
    Run simulations for different D values to compute front velocity.
    """
    print("\n" + "=" * 60)
    print("Running velocity sweep...")
    print("=" * 60)
    
    D_values = np.linspace(0.02, 0.3, 10)
    velocities = []
    theoretical_speeds = []
    
    for D in D_values:
        print(f"\n  D = {D:.3f}")
        model = ABM2D(nx=100, ny=100, D=D, beta=0.05, a=1.0, sigma=0.01)
        alpha0 = model.initialize_cluster(cluster_radius=10)
        results = model.run_simulation(alpha0, dt=0.01, T=300, save_snapshots=False)
        
        # Compute front velocity
        front_pos = results['front_positions']
        times = results['times']
        
        # Fit linear region
        idx = len(front_pos) // 4
        coeff = np.polyfit(times[idx:], front_pos[idx:], 1)
        velocity = coeff[0]
        velocities.append(velocity)
        
        # Theoretical speed
        theoretical = model.minimal_speed()
        theoretical_speeds.append(theoretical)
    
    return D_values, velocities, theoretical_speeds

def run_phase_transition_sweep():
    """
    Run phase transition sweep.
    """
    print("\n" + "=" * 60)
    print("Running phase transition sweep...")
    print("=" * 60)
    
    cluster_sizes = np.arange(2, 15, 1)
    success_rates = []
    n_runs_per_size = 20
    
    for size in cluster_sizes:
        print(f"\n  Cluster size: {size}")
        successes = 0
        
        for run in range(n_runs_per_size):
            model = ABM2D(nx=100, ny=100, D=0.1, beta=0.05, a=1.0, sigma=0.02)
            alpha0 = model.initialize_cluster(cluster_radius=size)
            results = model.run_simulation(alpha0, dt=0.01, T=500, save_snapshots=False)
            
            # Check if resistance spread
            final_mean = np.mean(results['alpha_final'])
            if final_mean > 1.0:
                successes += 1
        
        success_rates.append(successes / n_runs_per_size)
    
    # Find critical size (where success rate crosses 0.5)
    idx = np.where(np.array(success_rates) >= 0.5)[0]
    critical_size = cluster_sizes[idx[0]] if len(idx) > 0 else cluster_sizes[-1]
    
    return cluster_sizes, success_rates, critical_size

def main():
    """
    Main execution function.
    """
    # Run simulations
    results_2d = run_2d_simulation()
    velocity_results = run_velocity_sweep()
    phase_results = run_phase_transition_sweep()
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/velocity_results.pkl', 'wb') as f:
        pickle.dump(velocity_results, f)
    with open('data/phase_results.pkl', 'wb') as f:
        pickle.dump(phase_results, f)
    
    print("\n" + "=" * 60)
    print("Results saved.")
    print("=" * 60)
    
    # Generate figures
    print("\nGenerating figures...")
    fig_gen = FigureGenerator()
    # Note: network_results will be generated separately
    fig_gen.figure_wave_propagation(results_2d)
    fig_gen.figure_front_velocity(*velocity_results)
    fig_gen.figure_phase_transition(*phase_results)
    
    print("Done!")

if __name__ == "__main__":
    main()
