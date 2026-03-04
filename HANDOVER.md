# HANDOVER — OncoPrediction
## Document de passation pour Claude Code (VS Code)
---

## QUI JE SUIS
Étudiante en fin d'études de médecine humaine à l'Université de Genève.
Aucune compétence en programmation. Je dois pouvoir expliquer chaque décision
méthodologique devant un jury de master.

---

## MON PROJET
**Titre** : "Prédiction de la survie des patients lors des tumor boards avec
des cancers gastro-œsophagiens par intelligence artificielle"

**Deux livrables** :
1. Un modèle ML + application web interactive pour oncologues
2. Un article scientifique

**Données** : TCGA via cBioPortal — données publiques anonymisées
- 617 patients, cancers gastro-œsophagiens (440 estomac, 177 œsophage)
- 244 décès (39.5%), survie médiane 28.7 mois

---

## ENVIRONNEMENT TECHNIQUE
- **OS** : macOS (Darwin 23.1.0)
- **Python** : 3.9.6 (chemin : `/usr/bin/python3`)
- **pip** : `/Library/Developer/CommandLineTools/usr/bin/python3 -m pip`
- **Streamlit** : `/Users/alexandradubugnon/Library/Python/3.9/bin/streamlit`
- **Dossier projet** : `/Users/alexandradubugnon/Desktop/projet_master`
- **Pas de git** initialisé sur le projet local

### Packages installés (versions exactes)
```
numpy==1.26.4
pandas==2.3.3
matplotlib==3.9.4
scikit-learn==1.5.2
scikit-survival==0.23.1
lifelines==0.27.8      ← version spécifique ! (0.30.x ne marche pas sur Python 3.9)
joblib==1.5.3
openpyxl==3.1.5
streamlit==1.50.0
scipy==1.13.1
```

**⚠️ IMPORTANT** : `lifelines` DOIT rester en version `0.27.8`.
La version 0.30+ utilise `datetime.UTC` qui n'existe pas en Python 3.9.
Commande pour réinstaller si besoin : `pip3 install lifelines==0.27.8`

---

## STRUCTURE DU PROJET
```
projet_master/
├── data/
│   ├── CLEAN_dataset.xlsx          ← dataset propre (617 patients, 20 cols, 0 NaN)
│   ├── BRUT_donne_es_sans_pb_date.xlsx  ← données brutes originales
│   └── imputation_log.xlsx         ← log des imputations effectuées
├── scripts/
│   ├── 04_random_survival_forest.py    ← étape 3 (RSF)
│   ├── 05_model_comparison.py          ← étape 4 (Cox vs RSF)
│   └── save_deploy_models.py           ← prépare les modèles pour déploiement
├── models/
│   └── rsf_model.pkl               ← RSF 500 arbres, split 80/20 (86MB, local only)
├── figures/
│   ├── RSF_feature_importance.png
│   ├── RSF_survival_curves_profiles.png
│   ├── Comparaison_CIndex.png
│   ├── Calibration_Cox_vs_RSF.png
│   ├── BrierScore_Cox_vs_RSF.png
│   └── Tableau_Comparaison_Cox_RSF.png
├── results/
│   ├── RSF_resultats.xlsx
│   └── Comparaison_Cox_vs_RSF.xlsx
├── app/
│   └── app.py                      ← app Streamlit locale (utilise lifelines)
├── deploy/                         ← prêt pour Streamlit Community Cloud
│   ├── app.py                      ← app sans lifelines (sksurv uniquement)
│   ├── requirements.txt
│   ├── runtime.txt                 ← python-3.11
│   ├── .streamlit/config.toml
│   └── models/
│       ├── cox_model.pkl           ← Cox sksurv, 617 patients (8 KB)
│       └── rsf_model.pkl           ← RSF 200 arbres, 617 patients (4.1 MB)
├── huggingface_deploy/             ← prêt pour Hugging Face Spaces
│   ├── app.py                      ← identique à deploy/app.py
│   ├── requirements.txt            ← versions épinglées exactes
│   ├── README.md                   ← config YAML HF Spaces (sdk: streamlit)
│   └── models/
│       ├── cox_model.pkl
│       └── rsf_model.pkl
└── HANDOVER.md                     ← ce fichier
```

---

## CE QUI A ÉTÉ FAIT

### ✅ Étape 1 — Nettoyage des données (TERMINÉ, pas de script)
Le fichier `CLEAN_dataset.xlsx` est le résultat final. Décisions clés :
- Suppression des variables : identifiantes, data leakage, trop de manquants
- Encodages : OS_Status→OS_Event (0/1), Sex (0/1), Anatomic_Site (0/1)
- Stades TNM simplifiés (IIIA/B/C→III, T4A/B→T4, TX/NX/MX→NaN)
- Imputation : numérique→médiane, catégorielle→mode
- 5 patients supprimés (OS manquant) → 617 patients restants

### ✅ Étape 2 — Kaplan-Meier (TERMINÉ, pas de script conservé)
Résultats clés :
- Survie globale : médiane 28.7 mois, 1an=76.1%, 2ans=56.2%, 5ans=32.1%
- KM par stade AJCC : p<0.001 ✓ (I:72m, II:46m, III:21m, IV:13m)
- KM par grade : p=0.024 ✓
- KM par M0/M1 : p<0.001 ✓

### ✅ Étape 3 — Régression de Cox (TERMINÉ, pas de script conservé)
Modèle final raffiné — 6 variables :
| Variable | HR | IC 95% | p-value |
|---|---|---|---|
| Âge (par SD) | 1.213 | [1.061-1.388] | 0.005** |
| Grade (par SD) | 1.164 | [1.013-1.339] | 0.033* |
| Stade III (vs I-II) | 2.330 | [1.746-3.109] | <0.001*** |
| Stade IV (vs I-II) | 4.027 | [2.674-6.066] | <0.001*** |
| Radiothérapie | 0.451 | [0.310-0.657] | <0.001*** |
| TMB (par SD) | 0.738 | [0.591-0.921] | 0.007** |

**C-index = 0.683**
**⚠️ LIMITE** : hypothèse de proportionnalité des risques VIOLÉE pour 5/6
variables (test de Schoenfeld) → justifie l'usage du RSF

### ✅ Étape 4 — Random Survival Forest (script : `04_random_survival_forest.py`)
- Split : 80% train (493 patients) / 20% test (124 patients), stratifié sur OS_Event
- 44 features (one-hot encoding des catégorielles)
- Paramètres : n_estimators=500, min_samples_leaf=15, min_samples_split=10
- **C-index test set : 0.635**
- CV 5-fold : 0.641 ± 0.049
- Feature importance (permutation) : Stade AJCC III > Radiothérapie > Stade II > Âge > N-Stage N0
- Modèle sauvegardé : `models/rsf_model.pkl` (86 MB — trop grand pour GitHub)

### ✅ Étape 5 — Comparaison Cox vs RSF (script : `05_model_comparison.py`)
Sur le MÊME test set (124 patients) :
| Métrique | Cox | RSF | Meilleur |
|---|---|---|---|
| C-index | **0.665** | 0.635 | **Cox** |
| IBS (Integrated Brier Score) | **0.141** | 0.161 | **Cox** |

**CONCLUSION : Le Cox est meilleur sur ce dataset.**
Explication : dataset petit (617 patients), 6 variables bien sélectionnées
évitent le bruit des 44 features du RSF.

### ✅ Étape 6 — Application web Streamlit (app/app.py)
Fonctionnalités :
- Formulaire patient complet (sidebar) : clinique + histologie + génomique
- Courbes de survie Cox (trait plein) + RSF (pointillés)
- Métriques : survie 1/2/5 ans + médiane + niveau de risque
- Comparaison multi-patients (jusqu'à 6, session_state)
- Utilise lifelines 0.27.8 pour le Cox + sksurv pour le RSF

Pour lancer l'app locale :
```bash
/Users/alexandradubugnon/Library/Python/3.9/bin/streamlit run app/app.py
```

### ✅ Étape 7 — Préparation au déploiement
**deploy/** → Streamlit Community Cloud
**huggingface_deploy/** → Hugging Face Spaces (alexdddd1212/oncoprediction)

Différences entre app locale et versions déploiement :
- Pas de lifelines (remplacé par sksurv Cox → CoxPHSurvivalAnalysis)
- RSF allégé : 200 arbres au lieu de 500 (4.1 MB au lieu de 86 MB)
- Entraîné sur 100% des données (pas de split train/test)
- Stats des sliders intégrées dans les pkl (pas besoin du dataset)

---

---

## DOSSIER DEPLOY_TM — APPLICATION LIVE EN PRODUCTION

### Emplacement
```
/Users/alexandradubugnon/Desktop/deploy_tm/
├── app.py              ← VERSION EN PRODUCTION (avec auth Firebase)
├── requirements.txt
└── models/
    ├── cox_model.pkl   ← identiques à ceux de deploy/
    └── rsf_model.pkl
```

### Plateforme d'hébergement
- **Hébergeur** : Render (render.com) — pas Streamlit Cloud ni HF Spaces
- **Déploiement** : automatique via GitHub (push → rebuild automatique)
- **Repo GitHub** : connecté à Render, contient deploy_tm/

### Authentification Firebase
L'app `deploy_tm/app.py` contient un système d'authentification complet
via **Firebase Authentication** de Google (API REST directe, pas de SDK).

**Fonctionnalités implémentées** :
- Connexion email/mot de passe (`firebase_sign_in`)
- Inscription (`firebase_sign_up`)
- Réinitialisation de mot de passe par email (`firebase_reset_password`)
- Déconnexion (nettoyage du session_state)
- Badge utilisateur connecté affiché dans la sidebar

**Flux de l'app** :
```
Démarrage
    ↓
st.session_state["authenticated"] ?
    ├── Non → show_auth_page()  (connexion/inscription)
    └── Oui → main_app()       (formulaire patient + prédictions)
```

**Projet Firebase** :
- Nom du projet : `oncoprediction`
- Auth domain : `oncoprediction.firebaseapp.com`
- Méthode auth : Email/Password activée dans la console Firebase

### ⚠️ PROBLÈME DE SÉCURITÉ CRITIQUE — À CORRIGER EN PRIORITÉ
La clé API Firebase est actuellement **écrite en dur dans app.py**.
Si le repo GitHub est public, elle est visible par tout le monde.

**Ce qu'il faut faire dans VS Code (première tâche)** :
1. Supprimer la clé du code Python
2. La mettre dans une variable d'environnement Render :
   - Dans Render : Settings → Environment Variables → Add `FIREBASE_API_KEY`
3. Dans le code, remplacer par :
   ```python
   FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY")
   ```

### Différences entre deploy_tm/app.py et les autres versions
| Aspect | deploy_tm/ (LIVE) | deploy/ | huggingface_deploy/ |
|---|---|---|---|
| Auth Firebase | ✅ Oui | ❌ Non | ❌ Non |
| Hébergeur | Render | Streamlit Cloud | HF Spaces |
| Modèles | models/ (local) | models/ (local) | models/ (local) |
| Clé Firebase | ⚠️ En dur dans code | N/A | N/A |

### Requirements deploy_tm
```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0,<2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0,<1.6.0
scikit-survival>=0.21.0
joblib>=1.2.0
openpyxl>=3.1.0
requests>=2.28.0          ← pour les appels REST Firebase
```

---

## CE QUI RESTE À FAIRE

### 🔴 PRIORITÉ 1 — Sécuriser la clé Firebase dans deploy_tm/
Déplacer `FIREBASE_API_KEY` du code vers une variable d'environnement Render.
Voir section "PROBLÈME DE SÉCURITÉ CRITIQUE" ci-dessus.

### 🔲 Étape 8 — Figures pour l'article (scripts/06_figures_article.py)
- Tableau 1 (Table 1) : caractéristiques de la cohorte
- Courbe ROC time-dependent
- Courbe de calibration finale

### 🔲 Étape 9 — Déploiement HF Spaces
Le dossier `huggingface_deploy/` est prêt.
Username HF : alexdddd1212 | Space : oncoprediction
Manque : uploader via git ou interface web HF

### 🔲 Étape 10 — Article scientifique
Structure : Abstract / Introduction / Méthodes / Résultats / Discussion
Tous les chiffres sont disponibles dans results/ et dans ce document.

---

## VARIABLES DU DATASET (CLEAN_dataset.xlsx)

| Variable | Type | Description |
|---|---|---|
| Patient_ID | ID | Exclue du modèle |
| OS_Months | Numérique | Durée de suivi (cible 1) |
| OS_Event | Binaire | 0=Vivant, 1=Décédé (cible 2) |
| Age | Numérique | 27–90 ans, médiane=65 |
| Sex | Binaire | 0=Femme, 1=Homme |
| Anatomic_Site | Binaire | 0=Œsophage, 1=Estomac |
| Histology_Detailed | Catégorielle | 9 sous-types |
| Grade | Ordinal | 1/2/3 |
| AJCC_Stage_Simplified | Catégorielle | I/II/III/IV |
| T_Stage_Clean | Catégorielle | T1/T2/T3/T4 |
| N_Stage_Clean | Catégorielle | N0/N1/N2/N3 |
| M_Stage_Clean | Catégorielle | M0/M1 |
| Aneuploidy_Score | Numérique | 0–31 |
| Fraction_Genome_Altered | Numérique | 0–0.93 |
| MSIsensor_Score | Numérique | 0–45.66 |
| Mutation_Count | Numérique | 1–6363 |
| TMB | Numérique | 0–330.8 mut/Mb |
| Subtype | Catégorielle | 10 sous-types TCGA |
| Tumor_Break_Load | Numérique | 0–1518 |
| Radiation_Therapy | Binaire | 0=Non, 1=Oui |

---

## PIÈGES ET DÉCISIONS IMPORTANTES À CONNAÎTRE

1. **lifelines 0.27.8 OBLIGATOIRE** — ne pas upgrader (Python 3.9 incompatible avec 0.30+)
2. **numpy < 2.0** — les pkl ont été sauvegardés avec numpy 1.26.4
3. **scikit-learn 1.5.2** — scikit-survival 0.23.1 ne supporte pas 1.6+
4. **Le Cox doit toujours être entraîné avant utilisation** (app locale)
   ou chargé depuis pkl (versions déploiement)
5. **Streamlit** n'est pas dans le PATH → toujours utiliser le chemin complet
   `/Users/alexandradubugnon/Library/Python/3.9/bin/streamlit`
6. **Brier Score** : les temps d'évaluation doivent être < max(y_test['OS_Months'])
   Toujours appliquer un facteur 0.95 sur la borne maximale
7. **feature_importances_** du RSF → NotImplementedError dans sksurv
   → utiliser la permutation importance à la place
8. **Les pkl de déploiement** (dans deploy/ et huggingface_deploy/) sont différents
   du pkl local (models/rsf_model.pkl) :
   - Déploiement : 200 arbres, dataset complet, sksurv Cox inclus
   - Local : 500 arbres, split 80/20, uniquement RSF
