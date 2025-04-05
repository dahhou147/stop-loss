#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

def geometric_brownian_motion(S0, mu, sigma, N, T, M):
    """
    Génère des trajectoires de mouvement brownien géométrique
    
    Paramètres:
    S0: Prix initial
    mu: Taux de rendement
    sigma: Volatilité
    N: Nombre de pas de temps
    T: Horizon temporel
    M: Nombre de trajectoires
    
    Retourne:
    t: vecteur des temps
    S: matrice des prix de taille (N,M)
    """
    dt = T / N
    t = np.linspace(0, T, N)
    dW = ss.norm.rvs(scale=np.sqrt(dt), size=(N-1, M))
    W = np.cumsum(dW, axis=0)
    W = np.vstack([np.zeros(M), W])
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t[:, None] + sigma * W)
    return t, S

class BlackScholesSimulation:
    def __init__(self, S0, r, sigma, T, N):
        """
        Initialise la simulation du modèle de Black-Scholes
        Paramètres:
        S0: Prix initial de l'actif
        r: Taux sans risque
        sigma: Volatilité
        T: Horizon temporel
        dt: Pas de temps
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N
        self.dt = T/N
        self.t = np.arange(0, T + self.dt, self.dt)
    

    def simulate_paths(self, n_paths):
        """
        Simule n_paths trajectoires du prix de l'actif
        Retourne:
        - t: vecteur des temps
        - paths: matrice des prix de taille (N,M) où N est le nombre de pas de temps et M le nombre de trajectoires
        """
        t, paths = geometric_brownian_motion(self.S0, self.r, self.sigma, self.N, self.T, n_paths)
        return t, paths

    
    def plot_paths(self, n_paths=5):
        """
        Visualise les trajectoires simulées
        
        Paramètres:
        n_paths: Nombre de trajectoires à visualiser
        """
        t, paths = self.simulate_paths(n_paths)
        
        plt.figure(figsize=(10, 6))
        for i in range(n_paths):
            plt.plot(t, paths[:,i], label=f'Trajectoire {i+1}')
        
        plt.title('Simulation des trajectoires de prix selon Black-Scholes')
        plt.xlabel('Temps')
        plt.ylabel('Prix de l\'actif')
        plt.grid(True)
        plt.show()
#%%
# Exemple d'utilisation
if __name__ == "__main__":
    # Paramètres de simulation
    S0 = 100  # Prix initial
    r = 0.05  # Taux sans risque
    sigma = 0.2  # Volatilité
    T = 1  # Horizon temporel (1 an)
    N = 100  # Pas de temps
    
    # Création de l'instance et simulation
    simulation = BlackScholesSimulation(S0, r, sigma, T, N)
    simulation.plot_paths(1000) 
# %%
