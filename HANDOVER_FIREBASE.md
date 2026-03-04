# HANDOVER — Déploiement OncoPrediction + Firebase Auth

**Date** : 24 février 2026
**Auteur** : Documentation générée lors de la session Claude avec Alexandra
**Projet** : Mémoire de Master — Médecine, Université de Genève (HUG)

---

## 1. CONFIGURATION RENDER

### Service Web
- **URL du service** : https://oncoprediction.onrender.com
- **Domaine personnalisé** : https://oncoprediction.ch (vérifié + certificat SSL émis)
- **Domaine www** : https://www.oncoprediction.ch (redirige vers oncoprediction.ch)
- **Dashboard Render** : https://dashboard.render.com
- **Service ID** : srv-d6emblrh46gs73e5a7n0
- **Plan** : Free (le site s'endort après 15 min d'inactivité, ~30-60s pour se réveiller)
- **Région** : Frankfurt (EU Central)

### Paramètres de build et déploiement
- **Branch** : main
- **Root Directory** : `deploy`
- **Build Command** : `pip install -r requirements.txt`
- **Start Command** : `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- **Runtime** : Python 3
- **Auto-Deploy** : On Commit (se redéploie automatiquement quand on push sur GitHub)

### Variables d'environnement
- Aucune variable d'environnement configurée sur Render
- La configuration Firebase est directement dans le code `app.py` (les clés Firebase sont conçues pour être publiques)

---

## 2. REPO GITHUB

### Informations
- **Username** : alexandradubugnon-hash
- **Repo** : oncoprediction (public)
- **URL** : https://github.com/alexandradubugnon-hash/oncoprediction
- **Branch principale** : main

### Structure du repo
```
oncoprediction/
└── deploy/
    ├── app.py                 ← Application Streamlit avec auth Firebase
    ├── requirements.txt       ← Dépendances Python
    ├── runtime.txt            ← Version Python
    └── models/
        ├── cox_model.pkl      ← Modèle Cox sérializé (joblib)
        └── rsf_model.pkl      ← Modèle RSF sérializé (joblib)
```

### requirements.txt (version actuelle)
```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0,<2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0,<1.6.0
scikit-survival>=0.21.0
joblib>=1.2.0
openpyxl>=3.1.0
requests>=2.28.0
```
> `requests` a été ajouté pour l'authentification Firebase (API REST directe).

---

## 3. CONFIGURATION FIREBASE

### Projet Firebase
- **Nom du projet** : oncoprediction
- **Console** : https://console.firebase.google.com/project/oncoprediction
- **Plan** : Spark (gratuit)

### Application Web enregistrée
- **Nom** : oncoprediction-web
- **App ID** : 1:242172897083:web:2e2d211e86f97f79edf91e

### Configuration Firebase (firebaseConfig)
```javascript
const firebaseConfig = {
  apiKey: "AIzaSyALbtXNJEM47l71O1uFG7Z8a1FWvrEq3hY",
  authDomain: "oncoprediction.firebaseapp.com",
  projectId: "oncoprediction",
  storageBucket: "oncoprediction.firebasestorage.app",
  messagingSenderId: "242172897083",
  appId: "1:242172897083:web:2e2d211e86f97f79edf91e",
  measurementId: "G-09766X2VL1"
};
```

### Méthodes d'authentification activées
- **Email / Mot de passe** : ✅ Activé
- Autres méthodes (Google, etc.) : non activées

### Fonctionnalités d'auth implémentées dans l'app
- **Inscription** (création de compte email + mot de passe)
- **Connexion** (login email + mot de passe)
- **Mot de passe oublié** (envoi d'email de réinitialisation via Firebase)
- **Déconnexion** (bouton dans la sidebar)
- L'authentification utilise l'**API REST Firebase** directement (pas de SDK Python Firebase), ce qui évite les dépendances complexes comme pyrebase4

### Architecture de l'authentification dans app.py
```
main()
├── Si non authentifié → show_auth_page()
│   ├── Mode "login"  → formulaire de connexion
│   ├── Mode "signup"  → formulaire d'inscription
│   └── Mode "reset"   → formulaire mot de passe oublié
└── Si authentifié → main_app()
    └── Application OncoPrediction complète avec sidebar
```

Les fonctions clés :
- `firebase_sign_up(email, password)` → appelle `identitytoolkit.googleapis.com/v1/accounts:signUp`
- `firebase_sign_in(email, password)` → appelle `identitytoolkit.googleapis.com/v1/accounts:signInWithPassword`
- `firebase_reset_password(email)` → appelle `identitytoolkit.googleapis.com/v1/accounts:sendOobCode`

---

## 4. CONFIGURATION DNS (INFOMANIAK)

### Registrar
- **Fournisseur** : Infomaniak (infomaniak.com)
- **Domaine** : oncoprediction.ch
- **Expiration** : 22 février 2027
- **Propriétaire** : Alexandra Dubugnon
- **Dashboard** : https://manager.infomaniak.com

### Enregistrements DNS ajoutés
| Type | Source | Valeur | TTL |
|------|--------|--------|-----|
| **CNAME** | www.oncoprediction.ch | oncoprediction.onrender.com | 1 heure |
| **A** | oncoprediction.ch (@) | 216.24.57.1 | 1 heure |

> L'enregistrement AAAA n'a PAS été ajouté (pas nécessaire).

---

## 5. PROBLÈMES RENCONTRÉS ET SOLUTIONS

### Problème 1 : Premier déploiement Render échoue — "Root directory does not exist"
- **Cause** : Faute de frappe dans le Root Directory — `depoly` au lieu de `deploy`
- **Solution** : Corriger dans Settings → Root Directory → `deploy`, puis Manual Deploy

### Problème 2 : DNS non vérifié immédiatement sur Render
- **Cause** : Propagation DNS normale (prend du temps pour les domaines .ch)
- **Solution** : Attendre ~1 heure puis recliquer sur "Verify" dans Custom Domains
- **Note** : La propagation DNS peut prendre jusqu'à 24h dans certains cas

### Problème 3 : Firebase Auth — "API key not valid"
- **Cause** : Erreur de transcription de la clé API depuis la capture d'écran. Un "r" manquant dans la clé (`FWvEq3hY` au lieu de `FWvrEq3hY`)
- **Solution** : Corriger la clé dans app.py sur GitHub (remplacer `FWvEq3hY` par `FWvrEq3hY`)
- **Statut** : Correction communiquée à Alexandra, à vérifier que le commit a été fait

### ⚠️ POINT D'ATTENTION : Vérifier que la correction de la clé API a bien été appliquée
La dernière action demandée à Alexandra était de modifier la clé API dans `app.py` sur GitHub. Il faut vérifier que :
1. Le commit a bien été fait sur GitHub
2. Render a bien redéployé
3. La création de compte fonctionne sur oncoprediction.ch

---

## 6. PROCHAINES ÉTAPES RECOMMANDÉES

### 🔴 Prioritaire — À faire maintenant
1. **Vérifier que la correction de la clé API Firebase est en place** et que l'inscription/connexion fonctionne
2. **Tester le cycle complet** : créer un compte → se connecter → utiliser l'app → se déconnecter

### 🟡 Important — Avant la soutenance
3. **Personnalisation esthétique de l'app** (optionnel) : couleurs, logo, disposition
4. **Figures et tableaux pour l'article scientifique** :
   - Tableau 1 : Caractéristiques de la cohorte (Table 1 classique d'un article médical)
   - Courbe ROC time-dependent (si possible)
   - Courbe de calibration finale du meilleur modèle
   - Toutes les figures en 300 DPI
5. **Rédaction de l'article scientifique** : résumé, introduction, méthodes, résultats, discussion

### 🟢 Optionnel — Améliorations futures
6. **Astuce soutenance** : visiter oncoprediction.ch 1-2 min avant la présentation pour "réveiller" le serveur (plan gratuit Render)
7. **Upgrade Render** (7$/mois) si besoin d'un site toujours instantané
8. **Renouveler le domaine** oncoprediction.ch avant le 22 février 2027 sur Infomaniak

---

## 7. INFORMATIONS COMPLÉMENTAIRES

### Comptes et accès
| Service | URL | Username/Email |
|---------|-----|----------------|
| GitHub | github.com | alexandradubugnon-hash |
| Render | dashboard.render.com | (connecté via GitHub) |
| Firebase | console.firebase.google.com | (compte Google d'Alexandra) |
| Infomaniak | manager.infomaniak.com | Alexandra Dubugnon |
| Hugging Face | huggingface.co | alexdddd1212 (Space créé mais non utilisé) |

### Structure locale du projet (sur le Mac d'Alexandra)
```
~/Desktop/projet_master/
├── CLEAN_dataset.xlsx
├── BRUT_donne_es_sans_pb_date.xlsx
├── imputation_log.xlsx
├── data/
├── scripts/
│   ├── 04_random_survival_forest.py
│   └── 05_model_comparison.py
├── models/
│   ├── cox_model.pkl
│   └── rsf_model.pkl
├── figures/
├── results/
├── app/
│   └── app.py              ← version locale originale (SANS Firebase auth)
└── deploy/
    ├── app.py              ← version avec Firebase auth (la même que sur GitHub)
    ├── requirements.txt
    ├── runtime.txt
    └── models/
        ├── cox_model.pkl
        └── rsf_model.pkl
```

### Rappels importants pour le prochain Claude
- Alexandra est étudiante en médecine, elle **ne sait pas coder** → tout expliquer simplement, étape par étape
- Chaque décision doit être **justifiable devant un jury**
- La variable cible est toujours le couple **(OS_Months, OS_Event)** — analyse de survie avec censure
- Le modèle de **Cox est retenu comme meilleur modèle** (C-index 0.665 vs 0.635 pour le RSF)
- L'app utilise les **deux modèles** (Cox + RSF) en backend
- Le document de reprise complet du projet est dans la conversation et détaille toutes les étapes d'analyse (Kaplan-Meier, Cox, RSF, comparaison)

---

*Fin du document de handover — Dernière mise à jour : 24 février 2026*
