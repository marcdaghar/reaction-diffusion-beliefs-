#!/usr/bin/env python3
"""
Calibrate the model on social network data (Twitter, Facebook).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.abm_network import ABMNetwork

class SocialDataCalibrator:
    """
    Calibrate the ABM using social network data.
    """
    
    def __init__(self, data_dir='data/twitter_data'):
        self.data_dir = data_dir
        self.network = None
        self.initial_beliefs = None
    
    def load_network_from_edgelist(self, filename):
        """
        Load network from edge list.
        """
        self.network = nx.read_edgelist(f'{self.data_dir}/{filename}')
        print(f"Network loaded: {self.network.number_of_nodes()} nodes, "
              f"{self.network.number_of_edges()} edges")
        return self.network
    
    def infer_beliefs_from_text(self, texts, use_lda=True):
        """
        Infer initial beliefs from text data using topic modeling.
        """
        if use_lda:
            # LDA topic modeling
            vectorizer = CountVectorizer(max_features=1000, stop_words='english')
            doc_term_matrix = vectorizer.fit_transform(texts)
            
            lda = LatentDirichletAllocation(n_components=3, random_state=42)
            topic_dist = lda.fit_transform(doc_term_matrix)
            
            # Map topic to belief (assuming one topic represents resistance)
            # This would need calibration
            alpha = np.array([0, np.pi/4, np.pi/2])[np.argmax(topic_dist, axis=1)]
            self.initial_beliefs = alpha
        else:
            # Simple heuristic based on keyword presence
            keywords = ['debt', 'interest', 'bank', 'capitalism', 'crisis']
            scores = []
            for text in texts:
                score = sum([1 for k in keywords if k in text.lower()])
                scores.append(score)
            scores = np.array(scores)
            self.initial_beliefs = (scores / scores.max()) * np.pi/2
        
        print(f"Beliefs inferred: mean = {np.mean(self.initial_beliefs):.3f}, "
              f"std = {np.std(self.initial_beliefs):.3f}")
        return self.initial_beliefs
    
    def calibrate_parameters(self):
        """
        Calibrate model parameters from data.
        """
        # Network metrics
        if self.network:
            avg_degree = np.mean([d for n, d in self.network.degree()])
            clustering = nx.average_clustering(self.network)
            
            # Estimate beta from network density
            beta = 0.01 + 0.05 * (avg_degree / self.network.number_of_nodes())
            
            # Estimate D from clustering coefficient
            D = 0.05 + 0.1 * clustering
            
            # Estimate sigma from belief variance
            if self.initial_beliefs is not None:
                sigma = 0.01 + 0.05 * np.std(self.initial_beliefs)
            else:
                sigma = 0.02
            
            print(f"Calibrated parameters:")
            print(f"  beta = {beta:.4f}")
            print(f"  D = {D:.4f}")
            print(f"  sigma = {sigma:.4f}")
            
            return {
                'beta': beta,
                'D': D,
                'sigma': sigma
            }
        
        return None

def demo_twitter_data():
    """
    Demo using synthetic Twitter data.
    """
    print("=" * 60)
    print("Calibrating with synthetic social data...")
    print("=" * 60)
    
    # Generate synthetic Twitter data
    n_agents = 200
    texts = [
        "Interest rates are destroying the economy" if np.random.random() < 0.3
        else "The financial system is stable" 
        for _ in range(n_agents)
    ]
    
    # Add some variety
    for i in range(n_agents):
        if np.random.random() < 0.1:
            texts[i] = "Need to reform the monetary system. No more debt slavery."
        elif np.random.random() < 0.1:
            texts[i] = "The economy is growing. Markets are efficient."
    
    # Calibrate
    calibrator = SocialDataCalibrator()
    
    # Create synthetic network
    G = nx.watts_strogatz_graph(n_agents, 10, 0.1)
    calibrator.network = G
    
    # Infer beliefs
    calibrator.infer_beliefs_from_text(texts)
    
    # Calibrate parameters
    params = calibrator.calibrate_parameters()
    
    # Run simulation
    model = ABMNetwork(G, D=params['D'], beta=params['beta'], 
                      sigma=params['sigma'])
    results = model.run_simulation(calibrator.initial_beliefs, 
                                   dt=0.01, T=200, save_snapshots=False)
    
    print(f"\nSimulation with calibrated data:")
    print(f"  Initial mean alpha: {np.mean(calibrator.initial_beliefs):.3f}")
    print(f"  Final mean alpha: {np.mean(results['alpha_final']):.3f}")
    print(f"  Resistant agents: {np.sum(results['alpha_final'] > 1.0)} / {n_agents}")
    
    return results

def main():
    """
    Main calibration function.
    """
    results = demo_twitter_data()
    
    print("\n" + "=" * 60)
    print("Calibration complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
