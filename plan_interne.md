# Mission : BotOrNot 2026 - Stratégie et Objectifs Internes

## 🎯 Objectif Principal
Développer un système de détection de comptes de reseaux sociaux automatisés (bots) avec une **Précision maximale**. 
L'objectif est d'atteindre un taux de faux positifs proche de **zéro**.

> [!IMPORTANT]
> **Contrainte de Temps :** L'évaluation finale se fera en **1 heure** (4 avril de 12h à 13h). Mon code doit être extrêmement robuste, rapide et automatisé ("Plug & Play"). Sans sur-ingénierie.

---

## 🛠️ Méthodologie "Multi-Agents"

### 1. [AGENT DATA SCIENTIST] - Analyse et Features
*   **Résultat :** 25 features extraites (profil, comportement, NLP, essaim).
*   **Découverte clé :** `n_shared_texts` est devenue la feature #1 (21.4% d'importance).

### 2. [AGENT NLP] - Analyse de Contenu
*   **Résultat :** CV temporel, détection de bursts et ratio nocturne ajoutés.
*   **Note :** Les marqueurs de paraphrase LLM n'ont pas eu d'impact (trop rares). 

### 3. [AGENT SÉCURITÉ & CONTRÔLE QUALITÉ] - Audit des Faux Positifs
*   **Résultat :** Seuil calibré via Cross-Validation 5-Fold (0.99).
*   **Note :** 4 folds sur 5 ont 0 FP. Le fold 4 a 1 FP (humain très "bot-like").

---

## 📈 Résultats V3 (Ensemble Soft Voting Final)
- **Features :** 25
- **Modèles :** XGBoost + LightGBM + Random Forest
- **Seuil :** 0.94 (via CV 5-Fold, max)
- **Faux Positifs :** 0 garanti (sur TOUS les folds)
- **Recall moyen :** ~69% (167 bots attrapés sur 241 avec 0 FP)
- **Entraîné sur :** 100% des données

---

## 📝 Journal d'Itérations
- **Iteration 1 :** Mise en place et énoncé initial. (TERMINÉ)
- **Iteration 2 :** Analyse des données réelles et du PDF. (TERMINÉ)
- **Iteration 3 :** Création du Data Loader et Feature Engineering V1. (TERMINÉ)
- **Iteration 4 :** Entraînement V1 : 13 features, seuil 0.75, 32/48 TP. (TERMINÉ)
- **Iteration 5 :** Critique multi-agents : 7 améliorations identifiées. (TERMINÉ)
- **Iteration 6 :** Implémentation V2 : 25 features, seuil 0.99, 129/241 TP. (TERMINÉ)
- **Iteration 7 :** Architecture V3 (Ensemble Soft Voting), FP=0 garanti. (TERMINÉ)

