#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from black_scholes_simulation import BlackScholesSimulation

class StopLossStrategy:
    def __init__(self, initial_capital, r, sigma, rf, T, N, floor_percentage):
        """
        Initialise la stratégie Stop-Loss
        
        Paramètres:
        initial_capital: Capital initial
        r: Taux de rendement de l'actif risqué
        sigma: Volatilité
        rf: Taux sans risque
        T: Horizon temporel
        N: Nombre de pas de temps
        floor_percentage: Pourcentage du capital initial à garantir
        """
        self.initial_capital = initial_capital
        self.floor_percentage = floor_percentage
        self.floor_value = initial_capital * (floor_percentage / 100)
        self.rf = rf
        
        self.price_simulation = BlackScholesSimulation(initial_capital, r, sigma, T, N)
        
    def simulate_strategy(self, n_paths):
        """
        Simule la stratégie Stop-Loss sur n trajectoires
        
        Paramètres:
        n_paths: Nombre de trajectoires à simuler
        
        Retourne:
        Array contenant les valeurs du portefeuille pour chaque trajectoire
        """
        t, price_paths = self.price_simulation.simulate_paths(n_paths)
        n_steps, _ = price_paths.shape
        portfolio_value = np.zeros((n_steps, n_paths))
        
        # Calcul de la valeur minimale garantie (floor) pour chaque pas de temps
        time_grid = np.linspace(0, self.price_simulation.T, n_steps)
        floor_values = self.floor_value * np.exp(self.rf * time_grid)
        
        allocation = (price_paths > floor_values[:, None]).astype(float)
        portfolio_value = allocation * price_paths + (1 - allocation) * floor_values[:, None]
        
        return portfolio_value / self.initial_capital-1
    
    def plot_strategy(self, n_paths=10000):
        portfolio_values = self.simulate_strategy(n_paths)
        final_values = portfolio_values[-1]
        plt.figure(figsize=(10, 6))
        sns.histplot(final_values, bins=50, edgecolor='black',stat='density')
        sns.kdeplot(final_values, color='red')
        plt.axvline(x=self.floor_value/self.initial_capital-1, color='r', linestyle='--', 
                   label=f'Valeur minimale garantie ({self.floor_percentage}%)')
        plt.title('Distribution des valeurs finales du portefeuille')
        plt.xlabel('Valeur finale du portefeuille')
        plt.ylabel('Densité')
        plt.legend()
        plt.show()
        print(f"\nStatistiques sur {n_paths} simulations:")
        print(f"Valeur minimale garantie: {self.floor_percentage:.2f}%")
        print(f"moyenne de rendement finale: {np.mean(final_values):.2f}")
        print(f"médiane de rendement finale: {np.median(final_values):.2f}")
        print(f"Écart-type: {np.std(final_values):.2f}")
        print(f"Probabilité de tomber sous le plancher: {np.mean(final_values < self.floor_value/self.initial_capital)*100:.2f}%")


#%%
if __name__ == "__main__":
    # Paramètres
    initial_capital = 100000
    r = 0.08
    sigma = 0.2
    rf = 0.03
    T = 1
    N = 100
    floor_percentage = 95  # Garantir 95% du capital initial
    
    # Création et simulation de la stratégie
    strategy = StopLossStrategy(initial_capital, r, sigma, rf, T, N, floor_percentage)
    strategy.plot_strategy()
# %%
