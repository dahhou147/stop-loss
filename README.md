# Stratégie Stop-Loss avec Analyse de Sensibilité SHAP

Ce projet implémente une stratégie de gestion de portefeuille basée sur un stop-loss dynamique, avec une analyse de sensibilité utilisant les valeurs SHAP.

## Description

La stratégie implémentée permet de :
- Protéger un pourcentage minimum du capital initial (floor)
- Basculer vers l'actif sans risque lorsque le plancher est atteint
- Analyser l'impact des paramètres sur la performance via SHAP values

## Fonctionnalités

- Simulation de trajectoires de prix selon le modèle de Black-Scholes
- Implémentation de la stratégie Stop-Loss
- Analyse de sensibilité avec SHAP values
- Visualisation des résultats

## Structure du Projet

- `stop_loss_strategy.py` : Implémentation de la stratégie Stop-Loss
- `black_scholes_simulation.py` : Simulation des prix selon Black-Scholes
- `sensitvity.py` : Analyse de sensibilité avec SHAP
- `strategy_app.py` : Application principale

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/dahhou147/stop-loss-strategy.git
cd stop-loss-strategy
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

```python
from sensitivity import Sensitivity

# Initialisation
sensitivity = Sensitivity(initial_capital=100000)

# Analyse de sensibilité
sensitivity.plot_sensitivity(n_paths=1000, n_samples=100)

# Visualisation des rendements
sensitivity.plot_strategy_returns(rf=0.02, floor_percentage=95)
```

## Dépendances

- numpy>=1.24.3
- pandas>=2.0.3
- matplotlib>=3.7.1
- seaborn>=0.12.2
- scipy>=1.10.1
- shap>=0.45.1

## Auteur

- **Usama Dahhou** - [@dahhou147](https://github.com/dahhou147)

## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails. 