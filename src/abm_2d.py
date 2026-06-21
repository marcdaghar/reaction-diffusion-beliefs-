"""
Agent-Based Model on 2D grid for economic beliefs.
"""

import numpy as np
from tqdm import tqdm
from src.model import ReactionDiffusionModel

class ABM2D(ReactionDiffusionModel):
    """
    Agent-Based Model on 2D grid.
    Each agent is a cell in the grid.
    """
    
    def __init__(self, nx=100, ny=100, D=0.1, beta=0.05, a=1.0, sigma=0.02, epsilon=0.1):
        super().__init__(D, beta, a, sigma, epsilon)
        self.nx = nx
        self.ny = ny
        
    def initialize_random(self, p_resistant=0.05):
        """
        Initialize with random beliefs.
        
        Args:
            p_resistant: Probability of being resistant (α=π/2)
        """
        alpha = np.random.uniform(0, self.alpha_2, (self.nx, self.ny))
        # Add some resistant agents
        mask = np.random.random((self.nx, self.ny)) < p_resistant
        alpha[mask] = self.alpha_2
        return alpha
    
    def initialize_cluster(self, cluster_radius=5, cluster_center=None):
        """
        Initialize with a resistant cluster in the center.
        
        Args:
            cluster_radius: Radius of the resistant cluster
            cluster_center: (x, y) center of the cluster
        """
        if cluster_center is None:
            cluster_center = (self.nx // 2, self.ny // 2)
        
        alpha = np.zeros((self.nx, self.ny))
        
        # Create resistance cluster
        x0, y0 = cluster_center
        for i in range(self.nx):
            for j in range(self.ny):
                dist = np.sqrt((i - x0)**2 + (j - y0)**2)
                if dist < cluster_radius:
                    alpha[i, j] = self.alpha_2
        
        return alpha
    
    def run_simulation(self, alpha0, dt=0.01, T=1000, save_snapshots=True):
        """
        Run the ABM simulation.
        
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
        front_positions = []
        mean_alphas = []
        
        for step in tqdm(range(n_steps), desc="Running ABM"):
            t = step * dt
            
            # Update each agent
            for i in range(self.nx):
                for j in range(self.ny):
                    # Get neighbors (4-neighborhood)
                    neighbors = self._get_neighbors_2d(alpha, i, j, self.nx, self.ny)
                    
                    # Compute influence
                    influence = self.influence_field(alpha[i, j], neighbors)
                    
                    # Reaction term
                    reaction = self.reaction_term(alpha[i, j])
                    
                    # Noise
                    noise = self.sigma * np.random.randn() * np.sqrt(dt)
                    
                    # Update
                    dalpha = (reaction + self.beta * influence) * dt + noise
                    alpha[i, j] += dalpha
                    
                    # Clip
                    alpha[i, j] = np.clip(alpha[i, j], 0, self.alpha_2)
            
            # Track front position (resistant region boundary)
            front_pos = self._find_front_position(alpha)
            front_positions.append(front_pos)
            mean_alphas.append(np.mean(alpha))
            
            # Save snapshots
            if step in snapshot_indices:
                snapshots.append(alpha.copy())
                times.append(t)
        
        return {
            'alpha_final': alpha,
            'snapshots': np.array(snapshots),
            'times': np.array(times),
            'front_positions': np.array(front_positions),
            'mean_alphas': np.array(mean_alphas)
        }
    
    def _find_front_position(self, alpha):
        """
        Find the position of the front (boundary between α=0 and α=π/2).
        
        Returns:
            Front position (x-coordinate of the boundary)
        """
        # Find the median x-coordinate of resistant agents
        resistant = alpha > self.alpha_2 / 2
        if not np.any(resistant):
            return self.nx  # No resistance
        if np.all(resistant):
            return 0  # All resistant
        
        # Find boundary
        x_coords, y_coords = np.where(resistant)
        if len(x_coords) == 0:
            return self.nx
        
        # Use the median x-coordinate as front position
        return np.median(x_coords)
