# Simulation de Stratégie Stop-Loss

Ce projet implémente une simulation de stratégie de gestion de portefeuille avec stop-loss basée sur le modèle de Black-Scholes.

## Structure du projet

- `black_scholes_simulation.py` : Implémentation du modèle de Black-Scholes
- `stop_loss_strategy.py` : Implémentation de la stratégie Stop-Loss
- `sensitivity_analysis.py` : Analyse de sensibilité et étude des rendements
- `requirements.txt` : Dépendances du projet
- `michelin_data.csv` : Données historiques de l'action Michelin (à fournir)

## Installation

1. Créer un environnement virtuel Python :
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Simulation du modèle de Black-Scholes

```python
from black_scholes_simulation import BlackScholesSimulation

# Création d'une instance
simulation = BlackScholesSimulation(
    S0=100,      # Prix initial
    r=0.05,      # Taux sans risque
    sigma=0.2,   # Volatilité
    T=1,         # Horizon temporel
    dt=0.01      # Pas de temps
)

# Simulation et visualisation
simulation.plot_paths(5)  # Simule 5 trajectoires
```

### Simulation de la stratégie Stop-Loss

```python
from stop_loss_strategy import StopLossStrategy

# Création d'une instance
strategy = StopLossStrategy(
    initial_capital=100000,
    S0=100,
    r=0.08,
    sigma=0.2,
    rf=0.03,
    T=1,
    dt=0.01,
    floor_percentage=95  # Garantir 95% du capital initial
)

# Simulation et visualisation
strategy.plot_results(1000)  # Simule 1000 trajectoires
```

### Analyse de sensibilité

```python
from sensitivity_analysis import sensitivity_analysis, plot_sensitivity_results

# Exécution de l'analyse
results_df = sensitivity_analysis()

# Visualisation des résultats
plot_sensitivity_results(results_df)
```

## Données Michelin

Pour l'analyse des rendements de Michelin, vous devez fournir un fichier CSV contenant les prix de clôture de l'action. Le fichier doit contenir une colonne 'Close' avec les prix de clôture.

## Notes

- Les simulations sont basées sur le modèle de Black-Scholes qui suppose une distribution normale des rendements.
- La stratégie Stop-Loss implémentée est une version simplifiée qui ne prend en compte que deux états : actif risqué ou actif sans risque.
- Les résultats de l'analyse de sensibilité peuvent varier selon les paramètres choisis. 