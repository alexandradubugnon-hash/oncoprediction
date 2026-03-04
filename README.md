---
title: OncoPrediction
emoji: 🏥
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: Prédiction de survie — Cancers gastro-œsophagiens (Tumor Board)
---

# OncoPrediction 🏥

**Outil d'aide à la décision pour les Tumor Boards en oncologie digestive**

## Description

Application web de prédiction de la survie globale (*Overall Survival*) des patients atteints de **cancers gastro-œsophagiens**, développée dans le cadre d'un mémoire de Master en Médecine à l'Université de Genève.

Elle permet à un oncologue de saisir les caractéristiques cliniques et génomiques d'un patient et d'obtenir instantanément :
- 📈 Une **courbe de survie individuelle** (Cox + Random Survival Forest)
- 📊 Les probabilités de survie à **1 an, 2 ans, 5 ans**
- ⏱️ La **survie médiane estimée**
- 🚦 Une **catégorie de risque** (Faible / Intermédiaire / Élevé)
- 👥 La possibilité de **comparer plusieurs patients** côte à côte

## Données

- **Source** : TCGA (*The Cancer Genome Atlas*) via [cBioPortal](https://www.cbioportal.org/) — données publiques anonymisées
- **Cohorte** : 617 patients (440 cancers gastriques, 177 cancers œsophagiens)
- **Suivi médian** : 28,7 mois | **Décès observés** : 244 (39,5 %)

## Modèles

| Modèle | Variables | C-index (test set) | IBS |
|---|---|---|---|
| **Régression de Cox** *(référence)* | 6 | **0,665** | **0,141** |
| Random Survival Forest | 44 | 0,635 | 0,161 |

**Variables Cox** : âge, grade histologique, stade AJCC (III/IV vs I-II), radiothérapie, TMB

## ⚠️ Avertissement

Cet outil est destiné à la **recherche académique** et à l'aide à la décision.
Il **ne remplace pas** le jugement clinique d'un professionnel de santé.
Les prédictions sont issues d'un modèle entraîné sur des données nord-américaines (TCGA) et doivent être interprétées avec prudence.

## Auteur

Mémoire de Master en Médecine humaine — Université de Genève
Données : TCGA / cBioPortal (domaine public, CC BY 4.0)
