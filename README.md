# BotOrNot 2026 - Bot Detection System (V3 Ensemble)

Bienvenue sur le dépôt officiel de notre système de détection de bots pour la compétition **BotOrNot 2026**. 

## Notre Philosophie : La Précision avant tout

Dans cette compétition, nous avons pris pour règle d'or le barème des juges : **Mieux vaut rater un bot que de bannir un humain.** 

Notre système est donc conçu pour être un "détecteur de certitude". Nous ne cherchons pas à attraper tous les bots à tout prix, mais à identifier ceux dont le comportement est **indéniablement artificiel**.

##  L'Approche Technique (V3 Ensemble)

Plutôt que de parier sur un seul algorithme, nous utilisons une architecture de **Soft Voting Ensemble**. Le système interroge trois modèles différents et fait la moyenne de leurs probabilités :
1.  **XGBoost** (Le champion de la performance structurée)
2.  **LightGBM** (Pour capturer des patterns de données différents)
3.  **Random Forest** (Le garde-fou, pour la stabilité et éviter le sur-apprentissage)

### Nos Variables Clés ("Features")
Nous analysons 25 indicateurs par compte, notamment :
*   **L'analyse d'essaim (Swarm Detection) :** Nous comptons combien de textes identiques sont partagés entre des utilisateurs différents. C'est notre signal le plus puissant.
*   **La régularité mathématique :** Les bots postent souvent à des intervalles trop réguliers. Nous mesurons l'écart-type et le coefficient de variation du temps entre les posts.
*   **Les signatures de profil :** Ratio abonnés/abonnements, métadonnées du nom d'utilisateur, et z-score d'activité.

## Comment l'utiliser ?

### 1. Installation
Clonez le repo et installez les dépendances :
```powershell
pip install -r requirements.txt
```

### 2. Prédire (Le jour de la compétition)
Pour générer les prédictions à partir du dataset final du jury :
```powershell
python src/predict.py chemin/vers/le/dataset.json resultat.en.txt
```

## Performance & Robustesse (Optimisation Finale)
Notre modèle a été validé par une **Cross-Validation (5-Fold)** rigoureuse. Pour la phase finale, nous avons adopté une stratégie agressive mais sécurisée en fixant notre seuil de décision à 0.85. 

Ce réglage nous permet de maximiser la détection des bots tout en maintenant un risque de Faux Positif extrêmement faible, pour préserver l'intégrité des utilisateurs humains.

*   **Taux de Faux Positifs (Tests) : 0**
*   **Recall (Efficacité de détection) : ~75%** (selon nos derniers benchmarks sur les jeux de données de pratique).

---
*Soumission officielle de l'équipe SergeBotHunters pour BotOrNot 2026.*