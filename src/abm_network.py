"""
Agent-Based Model on complex networks.
"""

import numpy as np
import networkx as nx
from tqdm import tqdm
from src.model import ReactionDiffusionModel

class ABMNetwork(ReactionDiffusionModel):
    """
    Agent-Based Model on complex networks.
    """
    
    def __init__(self, network, D=0.1, beta=0.05, a=1.0, sigma=0.02, epsilon=0.1):
        super().__init__(D, beta, a, sigma, epsilon)
        self.network = network
        self.n_agents = network.number_of_nodes()
        self.adjacency = nx.to_numpy_array(network)
        
    def initialize_random(self, p_resistant=0.05):
        """
        Initialize with random beliefs.
        
        Args:
            p_resistant: Probability of being resistant
        """
        alpha = np.random.uniform(0, self.alpha_2, self.n_agents)
        mask = np.random.random(self.n_agents) < p_resistant
        alpha[mask] = self.alpha_2
        return alpha
    
    def initialize_cluster(self, cluster_size=10):
        """
        Initialize with a resistant cluster.
        
        Args:
            cluster_size: Number of initially resistant agents
        """
        alpha = np.zeros(self.n_agents)
        # Select random agents as resistant
        indices = np.random.choice(self.n_agents, cluster_size, replace=False)
        alpha[indices] = self.alpha_2
        return alpha
    
    def run_simulation(self, alpha0, dt=0.01, T=1000, save_snapshots=True):
        """
        Run the ABM simulation on the network.
        
        Returns:
            Dictionary with results
        """
        n_steps = int(T / dt)
        n_snapshots = 100
        
        if save_snapshots:
            snapshot_indices = np.linspace(0, n_steps-1, n_snapshots, dtype=int)
        else:
            snapshot_indices = []
        
        alpha = alpha0.copy()
        snapshots = []
        times = []
        mean_alphas = []
        resistant_counts = []
        
        for step in tqdm(range(n_steps), desc="Running network ABM"):
            t = step * dt
            
            # Update each agent
            for i in range(self.n_agents):
                # Get neighbors
                neighbors = np.where(self.adjacency[i] > 0)[0]
                if len(neighbors) == 0:
                    continue
                
                alpha_neighbors = alpha[neighbors]
                weights = self.adjacency[i][neighbors]
                
                # Compute influence
                influence = self.influence_field(alpha[i], alpha_neighbors, weights)
                
                # Reaction term
                reaction = self.reaction_term(alpha[i])
                
                # Noise
                noise = self.sigma * np.random.randn() * np.sqrt(dt)
                
                # Update
                dalpha = (reaction + self.beta * influence) * dt + noise
                alpha[i] += dalpha
                
                # Clip
                alpha[i] = np.clip(alpha[i], 0, self.alpha_2)
            
            # Track metrics
            mean_alphas.append(np.mean(alpha))
            resistant_counts.append(np.sum(alpha > self.alpha_2 / 2))
            
            # Save snapshots
            if step in snapshot_indices:
                snapshots.append(alpha.copy())
                times.append(t)
        
        return {
            'alpha_final': alpha,
            'snapshots': np.array(snapshots),
            'times': np.array(times),
            'mean_alphas': np.array(mean_alphas),
            'resistant_counts': np.array(resistant_counts)
        }
    
    @classmethod
    def create_er_network(cls, n_agents=1000, p=0.01):
        """Create Erdős-Rényi random network."""
        network = nx.erdos_renyi_graph(n_agents, p)
        return cls(network)
    
    @classmethod
    def create_ws_network(cls, n_agents=1000, k=10, p=0.1):
        """Create Watts-Strogatz small-world network."""
        network = nx.watts_strogatz_graph(n_agents, k, p)
        return cls(network)
    
    @classmethod
    def create_ba_network(cls, n_agents=1000, m=5):
        """Create Barabási-Albert scale-free network."""
        network = nx.barabasi_albert_graph(n_agents, m)
        return cls(network)
