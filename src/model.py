"""
Core reaction-diffusion model for economic beliefs.
Contains the continuous PDE and related functions.
"""

import numpy as np
from scipy.integrate import odeint

class ReactionDiffusionModel:
    """
    Continuous reaction-diffusion model for economic beliefs.
    
    Equation:
        ∂α/∂t = D ∇²α - V'(α) + β·I(α) + η
    
    State:
        α(x,t) ∈ [0, π/2] : behavioral angle
    """
    
    def __init__(self, D=0.1, beta=0.05, a=1.0, sigma=0.02, epsilon=0.1):
        """
        Args:
            D: Diffusion coefficient
            beta: Contagion intensity
            a: Potential barrier height
            sigma: Noise amplitude
            epsilon: Tanh scaling parameter
        """
        self.D = D
        self.beta = beta
        self.a = a
        self.sigma = sigma
        self.epsilon = epsilon
        
        # Attractors
        self.alpha_1 = 0.0
        self.alpha_2 = np.pi / 2.0
    
    def potential(self, alpha):
        """Bistable potential V(α) = a(α-α₁)²(α-α₂)²"""
        return self.a * (alpha - self.alpha_1)**2 * (alpha - self.alpha_2)**2
    
    def potential_gradient(self, alpha):
        """dV/dα = 2a α(α-α₂)(2α-α₂)"""
        return 2 * self.a * alpha * (alpha - self.alpha_2) * (2 * alpha - self.alpha_2)
    
    def influence_field(self, alpha_i, alpha_neighbors, weights=None):
        """
        Social influence field.
        
        Args:
            alpha_i: Current agent's angle
            alpha_neighbors: Array of neighbors' angles
            weights: Array of neighbor weights (optional)
        
        Returns:
            Scalar influence value
        """
        if weights is None:
            weights = np.ones_like(alpha_neighbors)
        
        diff = alpha_neighbors - alpha_i
        tanh_values = np.tanh(diff / self.epsilon)
        
        weighted_sum = np.sum(weights * tanh_values)
        total_weight = np.sum(weights)
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def reaction_term(self, alpha):
        """Reaction term: -dV/dα"""
        return -self.potential_gradient(alpha)
    
    def lambda_alpha(self, alpha):
        """Linearized growth rate at equilibrium"""
        # Approximate Fisher-KPP growth rate
        return self.beta * alpha * (1 - alpha / self.alpha_2)
    
    def minimal_speed(self):
        """Theoretical minimal wave speed: c* = 2√(D·β)"""
        return 2 * np.sqrt(self.D * self.beta)
    
    def laplacian_2d(self, alpha_grid):
        """
        Compute 2D Laplacian with periodic boundary conditions.
        
        Args:
            alpha_grid: 2D array (nx, ny)
        
        Returns:
            Laplacian: 2D array
        """
        nx, ny = alpha_grid.shape
        lap = np.zeros_like(alpha_grid)
        
        # Periodic boundaries
        lap[1:-1, :] += alpha_grid[2:, :] + alpha_grid[:-2, :] - 2 * alpha_grid[1:-1, :]
        lap[:, 1:-1] += alpha_grid[:, 2:] + alpha_grid[:, :-2] - 2 * alpha_grid[:, 1:-1]
        
        # Boundary conditions (periodic)
        lap[0, :] += alpha_grid[1, :] + alpha_grid[-1, :] - 2 * alpha_grid[0, :]
        lap[-1, :] += alpha_grid[0, :] + alpha_grid[-2, :] - 2 * alpha_grid[-1, :]
        lap[:, 0] += alpha_grid[:, 1] + alpha_grid[:, -1] - 2 * alpha_grid[:, 0]
        lap[:, -1] += alpha_grid[:, 0] + alpha_grid[:, -2] - 2 * alpha_grid[:, -1]
        
        return lap / 4  # Normalize for 4-neighbor Laplacian
    
    def simulate_2d(self, alpha0, nx, ny, dt, T, verbose=True):
        """
        Simulate the reaction-diffusion equation on a 2D grid.
        
        Args:
            alpha0: Initial condition (2D array)
            nx, ny: Grid dimensions
            dt: Time step
            T: Total simulation time
            verbose: Print progress
        
        Returns:
            Dictionary with time series
        """
        n_steps = int(T / dt)
        n_snapshots = min(100, n_steps // 10)
        snapshot_indices = np.linspace(0, n_steps-1, n_snapshots, dtype=int)
        
        alpha = alpha0.copy()
        snapshots = []
        times = []
        energies = []
        mean_alpha = []
        
        for step in range(n_steps):
            t = step * dt
            
            # Compute Laplacian
            lap = self.laplacian_2d(alpha)
            
            # Compute potential gradient
            grad_V = self.potential_gradient(alpha)
            
            # Compute influence (simplified: mean field)
            # For 2D grid, we use local neighborhood average
            influence = np.zeros_like(alpha)
            for i in range(nx):
                for j in range(ny):
                    neighbors = self._get_neighbors_2d(alpha, i, j, nx, ny)
                    influence[i, j] = self.influence_field(alpha[i, j], neighbors)
            
            # Noise
            noise = self.sigma * np.random.randn(nx, ny) * np.sqrt(dt)
            
            # Update: α(t+dt) = α(t) + [D∇²α - V'(α) + β·I]dt + η√dt
            dalpha = (self.D * lap - grad_V + self.beta * influence) * dt + noise
            alpha += dalpha
            
            # Clip to [0, π/2]
            alpha = np.clip(alpha, 0, self.alpha_2)
            
            # Store snapshots
            if step in snapshot_indices:
                snapshots.append(alpha.copy())
                times.append(t)
                energies.append(self._total_energy(alpha))
                mean_alpha.append(np.mean(alpha))
            
            if verbose and step % (n_steps // 10) == 0:
                print(f"Step {step}/{n_steps}, mean α = {np.mean(alpha):.3f}")
        
        return {
            'alpha': alpha,
            'snapshots': np.array(snapshots),
            'times': np.array(times),
            'energies': np.array(energies),
            'mean_alpha': np.array(mean_alpha)
        }
    
    def _get_neighbors_2d(self, alpha, i, j, nx, ny):
        """Get 4-neighbors with periodic boundary conditions."""
        neighbors = []
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni = (i + di) % nx
            nj = (j + dj) % ny
            neighbors.append(alpha[ni, nj])
        return np.array(neighbors)
    
    def _total_energy(self, alpha):
        """Compute total energy: ∫V(α) + (D/2)|∇α|²"""
        nx, ny = alpha.shape
        energy = np.sum(self.potential(alpha))
        
        # Gradient energy (discrete)
        grad_x = np.diff(alpha, axis=0, append=alpha[0:1, :])
        grad_y = np.diff(alpha, axis=1, append=alpha[:, 0:1])
        energy += self.D/2 * (np.sum(grad_x**2) + np.sum(grad_y**2))
        
        return energy
