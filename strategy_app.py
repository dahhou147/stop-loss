#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from stop_loss_strategy import StopLossStrategy


        

def load_michelin_data():
    """
    Charge et prépare les données de Michelin
    
    Paramètres:
    ticker: Ticker de Michelin
    """
    data = pd.read_csv('michelin.csv')
    return data

def plot_return_distribution(title="Distribution des rendements"):
    """
    Trace la distribution empirique des rendements et la densité gaussienne
    
    Paramètres:
    returns: Série des rendements
    title: Titre du graphique
    """
    plt.figure(figsize=(12, 6))
    data = load_michelin_data()['Close']
    returns = np.log(data/data.shift(1)).dropna()
    sns.histplot(returns, stat='density', alpha=0.8, label='Distribution empirique')
    
    kde = gaussian_kde(returns)
    x_range = np.linspace(returns.min(), returns.max(), 100)
    plt.plot(x_range, kde(x_range), 'r-', label='Densité estimée (KDE)')
    # Densité gaussienne théorique
    mu = returns.mean()
    sigma = returns.std()
    gaussian = np.exp(-(x_range - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    plt.plot(x_range, gaussian, 'g--', label='Densité gaussienne')
    
    plt.title(title)
    plt.xlabel('Rendements')
    plt.ylabel('Densité')
    plt.legend()
    plt.grid(True)
    plt.show()

def sensitivity_analysis(rf,floor_percentage):
    data = load_michelin_data()["Close"]
    returns = np.log(data/data.shift(1))
    mu = returns.mean()*252
    sigma = returns.std()*np.sqrt(252)
    strategy = StopLossStrategy(initial_capital=100000, r=mu, sigma=sigma, rf=rf, T=1, N=252, floor_percentage=floor_percentage)
    portfolio_values = strategy.simulate_strategy(1000)
    return portfolio_values[-1],sigma


def plot_strategy_returns(rf=0.02, floor_percentage=95, n_simulations=1000):
    """
    Trace la distribution des rendements de la stratégie stop-loss
    
    Paramètres:
    rf: Taux sans risque
    floor_percentage: Pourcentage du capital initial à garantir
    n_simulations: Nombre de simulations à effectuer
    """
    final_values,sigma = sensitivity_analysis(rf, floor_percentage)
    plt.figure(figsize=(12, 6))
    
    sns.histplot(final_values, stat='density', alpha=0.7, label='Distribution des rendements')
    
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
    
    print(f"\nStatistiques sur {n_simulations} simulations:")
    print(f"Rendement moyen: {np.mean(final_values)*100:.2f}%")
    print(f"Rendement médian: {np.median(final_values)*100:.2f}%")
    print(f"Écart-type: {np.std(final_values)*100:.2f}%")
    print(f"Probabilité de tomber sous le plancher: {np.mean(final_values < floor_percentage/100)*100:.2f}%")


#%%
if __name__ == "__main__":
    results_df = sensitivity_analysis(0.02,0.95)
    plot_return_distribution()
    plot_strategy_returns()
# %%
