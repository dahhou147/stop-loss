#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stop_loss_strategy import StopLossStrategy
from strategy_app import load_michelin_data
from scipy.stats import gaussian_kde
import shap


class Sensitivity:
    def __init__(self, initial_capital):
        """
        Initialise l'analyse de sensibilité
        
        Paramètres:
        initial_capital: Capital initial
        """
        self.model = StopLossStrategy
        self.data = load_michelin_data()
        self.initial_capital = initial_capital
        
        # Calcul du rendement moyen annuel et de la volatilité
        returns = np.log(self.data['Close']/self.data['Close'].shift(1))
        self.r = returns.mean() * 252  # Rendement moyen annualisé
        self.sigma = returns.std() * np.sqrt(252)  # Volatilité annualisée
        self.T = 1  # Horizon temporel (1 an)
        self.N = 252  # Nombre de pas de temps (un par jour)

    def get_simulation(self, rf, floor_percentage, n_paths=1000):
        """
        Simule la stratégie Stop-Loss avec les paramètres donnés
        
        Paramètres:
        rf: Taux sans risque
        floor_percentage: Pourcentage du capital initial à garantir
        n_paths: Nombre de trajectoires à simuler
        
        Retourne:
        Valeurs finales du portefeuille et la volatilité
        """
        strategy = self.model(
            self.initial_capital,
            self.r,
            self.sigma,
            rf,
            self.T,
            self.N,
            floor_percentage
        )
        portfolio_values = strategy.simulate_strategy(n_paths)
        return portfolio_values[-1], self.sigma

    def plot_sensitivity(self, n_paths=1000, n_samples=100):
        """
        Visualise l'analyse de sensibilité des paramètres avec SHAP values
        
        Paramètres:
        n_paths: Nombre de trajectoires par simulation
        n_samples: Nombre d'échantillons pour l'analyse
        """
        # Création des échantillons de paramètres
        np.random.seed(42)
        rf_samples = np.random.uniform(0.01, 0.05, n_samples)  # Taux sans risque entre 1% et 5%
        floor_samples = np.random.uniform(90, 98, n_samples)  # Floor entre 90% et 98%
        sigma_samples = np.random.uniform(0.1, 0.5, n_samples)  # Volatilité entre 10% et 50%
        
        # Préparation des données pour SHAP
        X = np.column_stack([rf_samples, floor_samples, sigma_samples])
        feature_names = ['Taux sans risque', 'Floor %', 'Volatilité']
        
        # Calcul des valeurs finales pour chaque combinaison
        final_values = np.zeros(n_samples)
        for i in range(n_samples):
            portfolio_values, _ = self.get_simulation(
                rf_samples[i],
                floor_samples[i],
                n_paths
            )
            final_values[i] = np.mean(portfolio_values)  # On prend la moyenne des valeurs finales
        
        # Calcul des valeurs SHAP
        explainer = shap.KernelExplainer(
            lambda x: self._predict_shap(x, n_paths),
            X
        )
        shap_values = explainer.shap_values(X)
        
        # Création des graphiques SHAP
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            show=False
        )
        plt.title('Impact des paramètres sur la valeur finale moyenne du portefeuille')
        plt.tight_layout()
        plt.show()
        
        # Graphiques de dépendance
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
        print(f"Nombre de trajectoires par échantillon: {n_paths}")
        print("\nImpact moyen des paramètres:")
        for i, feature in enumerate(feature_names):
            print(f"{feature}: {np.abs(shap_values[:, i]).mean():.4f}")
        print(f"\nVolatilité de base: {self.sigma:.4f}")

    def _predict_shap(self, X, n_paths):
        """
        Fonction auxiliaire pour SHAP qui prédit les valeurs finales moyennes
        """
        predictions = np.zeros(len(X))
        for i, (rf, floor, sigma) in enumerate(X):
            portfolio_values, _ = self.get_simulation(rf, floor, n_paths)
            predictions[i] = np.mean(portfolio_values)  # On prend la moyenne des valeurs finales
        return predictions

    def plot_strategy_returns(self, rf=0.02, floor_percentage=95, n_simulations=1000):
        """
        Trace la distribution des rendements de la stratégie stop-loss
        
        Paramètres:
        rf: Taux sans risque
        floor_percentage: Pourcentage du capital initial à garantir
        n_simulations: Nombre de simulations à effectuer
        """
        final_values, sigma = self.get_simulation(rf, floor_percentage, n_simulations)
        
        plt.figure(figsize=(12, 6))
        
        # Histogramme des rendements avec seaborn
        sns.histplot(final_values, stat='density', alpha=0.7, label='Distribution des rendements')
        
        # Estimation de la densité par noyau gaussien
        kde = gaussian_kde(final_values)
        x_range = np.linspace(min(final_values), max(final_values), 100)
        plt.plot(x_range, kde(x_range), 'r-', label='Densité estimée (KDE)')
        
        plt.axvline(x=(floor_percentage/100-1), color='g', linestyle='--', 
                    label=f'Plancher de protection ({floor_percentage}%)')
        
        plt.title(f'Distribution des rendements de la stratégie Stop-Loss\n(rf={rf}, σ={sigma:.2f}, protection={floor_percentage}%)')
        plt.xlabel('Rendement')
        plt.ylabel('Densité')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Statistiques descriptives
        print(f"\nStatistiques sur {n_simulations} simulations:")
        print(f"Rendement moyen: {np.mean(final_values)*100:.2f}%")
        print(f"Rendement médian: {np.median(final_values)*100:.2f}%")
        print(f"Écart-type: {np.std(final_values)*100:.2f}%")
        print(f"Probabilité de tomber sous le plancher: {np.mean(final_values < floor_percentage/100)*100:.2f}%")

# Exemple d'utilisation
if __name__ == "__main__":
    sensitivity = Sensitivity(initial_capital=100000)
    sensitivity.plot_sensitivity(n_paths=1000, n_samples=500)
    sensitivity.plot_strategy_returns(rf=0.02, floor_percentage=95)
#%%
