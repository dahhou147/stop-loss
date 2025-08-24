import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from black_scholes_simulation import BlackScholesSimulation
import shap

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
        n_steps, n_paths = price_paths.shape
        portfolio_value = np.zeros((n_steps, n_paths))
        
        time_grid = np.linspace(0, self.price_simulation.T, n_steps)
        floor_values = self.floor_value * np.exp(self.rf * time_grid)
        
        allocation = (price_paths > floor_values[:, None]).astype(float)
        portfolio_value = allocation * price_paths + (1 - allocation) * floor_values[:, None]
        
        return portfolio_value / self.initial_capital

    def plot_strategy(self):
        """
        Visualise la distribution des valeurs finales du portefeuille
        """
        portfolio_values = self.simulate_strategy(1000)
        final_values = portfolio_values[-1, :]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(final_values, kde=True)
        plt.axvline(self.floor_percentage/100, color='red', linestyle='--', 
                   label=f'Floor ({self.floor_percentage}%)')
        plt.title('Distribution des valeurs finales du portefeuille')
        plt.xlabel('Valeur finale / Capital initial')
        plt.ylabel('Fréquence')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print("Statistiques sur les valeurs finales:")
        print(f"Moyenne: {np.mean(final_values):.2%}")
        print(f"Écart-type: {np.std(final_values):.2%}")
        print(f"Minimum: {np.min(final_values):.2%}")
        print(f"Maximum: {np.max(final_values):.2%}")
        print(f"Probabilité de tomber sous le plancher: {np.mean(final_values < self.floor_value/self.initial_capital)*100:.2f}%")

    def calculate_shap_values(self, n_samples=1000):
        """
        Calcule les valeurs SHAP pour les paramètres de la stratégie
        
        Paramètres:
        n_samples: Nombre d'échantillons pour l'analyse SHAP
        
        Retourne:
        Valeurs SHAP et les paramètres correspondants
        """
        np.random.seed(42)
        rf_samples = np.random.uniform(0.01, 0.05, n_samples)  # Taux sans risque entre 1% et 5%
        sigma_samples = np.random.uniform(0.15, 0.25, n_samples)  # Volatilité entre 15% et 25%
        floor_samples = np.random.uniform(90, 98, n_samples)  # Floor entre 90% et 98%
        
        final_values = np.zeros(n_samples)
        for i in range(n_samples):
            strategy = StopLossStrategy(
                self.initial_capital,
                self.price_simulation.r,
                sigma_samples[i],
                rf_samples[i],
                self.price_simulation.T,
                self.price_simulation.N,
                floor_samples[i]
            )
            portfolio_values = strategy.simulate_strategy(1)
            final_values[i] = portfolio_values[-1, 0]
        
        X = np.column_stack([rf_samples, sigma_samples, floor_samples])
        feature_names = ['Taux sans risque', 'Volatilité', 'Floor %']
        
        explainer = shap.KernelExplainer(lambda x: self._predict_shap(x, n_samples), X)
        shap_values = explainer.shap_values(X)
        
        return shap_values, X, feature_names, final_values
    
    def _predict_shap(self, X, n_samples):
        """
        Fonction auxiliaire pour SHAP qui prédit les valeurs finales
        """
        predictions = np.zeros(len(X))
        for i, (rf, sigma, floor) in enumerate(X):
            strategy = StopLossStrategy(
                self.initial_capital,
                self.price_simulation.r,
                sigma,
                rf,
                self.price_simulation.T,
                self.price_simulation.N,
                floor
            )
            portfolio_values = strategy.simulate_strategy(1)
            predictions[i] = portfolio_values[-1, 0]
        return predictions
    
    def plot_shap_analysis(self, n_samples=1000):
        """
        Visualise l'analyse SHAP des paramètres
        """
        shap_values, X, feature_names, final_values = self.calculate_shap_values(n_samples)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            show=False
        )
        plt.title('Impact des paramètres sur la valeur finale du portefeuille')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(15, 5))
        for i, feature in enumerate(feature_names):
            plt.subplot(1, 3, i+1)
            shap.dependence_plot(i, shap_values, X, feature_names=feature_names)
            plt.title(f'Dépendance SHAP pour {feature}')
        plt.tight_layout()
        plt.show()
        
        # Affichage des statistiques
        print("\nStatistiques sur l'analyse SHAP:")
        print(f"Nombre d'échantillons: {n_samples}")
        print("\nImpact moyen des paramètres:")
        for i, feature in enumerate(feature_names):
            print(f"{feature}: {np.abs(shap_values[:, i]).mean():.4f}")
if __name__ == "__main__":
    # Paramètres
    initial_capital = 100000
    r = 0.08
    sigma = 0.2
    rf = 0.03
    T = 1
    N = 100
    floor_percentage = 95
    
    # Création et simulation de la stratégie
    strategy = StopLossStrategy(initial_capital, r, sigma, rf, T, N, floor_percentage)
    
    # Affichage de la distribution initiale
    strategy.plot_strategy()
    
    # Analyse SHAP
    strategy.plot_shap_analysis(n_samples=100)
