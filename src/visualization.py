"""
Visualization functions for the reaction-diffusion article.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib import rcParams
from matplotlib.patches import Circle

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300

class FigureGenerator:
    """
    Generate figures for the reaction-diffusion article.
    """
    
    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def figure_wave_propagation(self, results, nx=100, ny=100):
        """
        Figure 1: Wave propagation (4-panel animation).
        """
        snapshots = results['snapshots']
        times = results['times']
        
        # Select 4 equidistant snapshots
        n_snapshots = len(snapshots)
        indices = np.linspace(0, n_snapshots-1, 4, dtype=int)
        selected_snapshots = snapshots[indices]
        selected_times = times[indices]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (ax, alpha, t) in enumerate(zip(axes, selected_snapshots, selected_times)):
            im = ax.imshow(alpha, cmap='RdBu_r', vmin=0, vmax=np.pi/2, 
                          extent=[0, nx, 0, ny], origin='lower')
            ax.set_title(f'$t = {t:.1f}$ years', fontsize=12)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(False)
            
            # Add colorbar
            if idx == 3:
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label(r'$\alpha(x,t)$', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/wave_propagation.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/wave_propagation.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_front_velocity(self, D_values, velocities, theoretical_speeds):
        """
        Figure 2: Front velocity vs D.
        
        Args:
            D_values: Array of diffusion coefficients
            velocities: Simulated front velocities
            theoretical_speeds: Theoretical minimal speeds
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot simulated velocities
        ax.plot(D_values, velocities, 'o-', linewidth=2.5, markersize=8, 
                color='darkblue', label='Simulated velocity')
        
        # Plot theoretical prediction
        ax.plot(D_values, theoretical_speeds, '--', linewidth=2.5, 
                color='red', label=r'$c^* = 2\sqrt{D\beta}$')
        
        ax.set_xlabel(r'Diffusion coefficient $D$', fontsize=12)
        ax.set_ylabel(r'Front velocity $c$ (lattice units/year)', fontsize=12)
        ax.set_title('Front Velocity as Function of Diffusion Coefficient', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/front_velocity.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/front_velocity.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_phase_transition(self, cluster_sizes, success_rates, critical_size):
        """
        Figure 3: Phase transition.
        
        Args:
            cluster_sizes: Array of initial cluster sizes
            success_rates: Probability of transition (success)
            critical_size: Estimated critical cluster size
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot success rate
        ax.plot(cluster_sizes, success_rates, 'o-', linewidth=2.5, markersize=8,
                color='darkgreen', label='Simulation')
        
        # Mark critical size
        ax.axvline(x=critical_size, color='red', linestyle='--', linewidth=2,
                   label=f'Critical size: $R_c = {critical_size:.1f}$')
        
        # Fit sigmoid
        from scipy.optimize import curve_fit
        def sigmoid(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))
        
        popt, _ = curve_fit(sigmoid, cluster_sizes, success_rates, maxfev=5000)
        x_smooth = np.linspace(cluster_sizes[0], cluster_sizes[-1], 100)
        y_smooth = sigmoid(x_smooth, *popt)
        ax.plot(x_smooth, y_smooth, ':', linewidth=2, color='orange', label='Fit')
        
        ax.set_xlabel(r'Initial cluster radius $R_0$ (lattice units)', fontsize=12)
        ax.set_ylabel('Probability of successful transition', fontsize=12)
        ax.set_title('Phase Transition: Critical Cluster Size', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, cluster_sizes[-1] * 1.1])
        ax.set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/phase_transition.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/phase_transition.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_network_topology(self, network_results):
        """
        Figure 4: Network topology comparison.
        
        Args:
            network_results: Dictionary with results for different topologies
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel 1: Evolution of mean alpha
        ax = axes[0, 0]
        colors = {'ER': 'blue', 'WS': 'green', 'BA': 'red'}
        for name, result in network_results.items():
            times = result['times']
            mean_alphas = result['mean_alphas']
            ax.plot(times, mean_alphas, '-', linewidth=2, color=colors[name], 
                    label=name)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel(r'$\langle \alpha \rangle$')
        ax.set_title('Mean Belief Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Resistant count evolution
        ax = axes[0, 1]
        for name, result in network_results.items():
            times = result['times']
            resistant_counts = result['resistant_counts']
            ax.plot(times, resistant_counts / 1000, '-', linewidth=2, 
                    color=colors[name], label=name)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel(r'Resistant agents ($\times 10^3$)')
        ax.set_title('Resistance Spread')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Network visualization (BA as example)
        ax = axes[1, 0]
        G = nx.barabasi_albert_graph(200, 5)
        pos = nx.spring_layout(G, seed=42)
        ax.set_title('Scale-free Network (BA)')
        nx.draw(G, pos, ax=ax, node_size=10, with_labels=False, 
                node_color='gray', edge_color='lightgray')
        ax.set_aspect('equal')
        
        # Panel 4: Final alpha distribution
        ax = axes[1, 1]
        for name, result in network_results.items():
            alpha_final = result['alpha_final']
            ax.hist(alpha_final, bins=30, alpha=0.5, label=name, density=True)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('Density')
        ax.set_title('Final Belief Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/network_topology.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/network_topology.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_all(self, results_2d, velocity_results, phase_results, network_results):
        """
        Generate all figures.
        """
        print("Generating Figure 1: Wave propagation...")
        self.figure_wave_propagation(results_2d)
        
        print("Generating Figure 2: Front velocity...")
        D_values, velocities, theoretical_speeds = velocity_results
        self.figure_front_velocity(D_values, velocities, theoretical_speeds)
        
        print("Generating Figure 3: Phase transition...")
        cluster_sizes, success_rates, critical_size = phase_results
        self.figure_phase_transition(cluster_sizes, success_rates, critical_size)
        
        print("Generating Figure 4: Network topology...")
        self.figure_network_topology(network_results)
        
        print("All figures generated successfully!")
