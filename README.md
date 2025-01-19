# meshia

```bash
text_to_mesh/
├── app/
│   ├── api.py               # API FastAPI pour les endpoints
│   ├── schemas.py           # Définitions des schémas de données pour l'API
│   ├── __init__.py          # Fichier d'initialisation pour le module FastAPI
├── main.py                  # Script pour démarrer l'application FastAPI
├── models/
│   ├── text_encoder.py      # Classe pour l'encodage des descriptions textuelles
│   ├── image_encoder.py     # Classe pour l'encodage des images
│   ├── point_decoder.py     # Modèle génératif pour le nuage de points
│   └── __init__.py
├── utils/
│   ├── data_utils.py        # Fonctions utilitaires pour la gestion des données
│   ├── visualization.py     # Outils pour visualiser les nuages de points et maillages
│   └── __init__.py
├── configs/
│   └── config.json          # Fichier de configuration pour les hyperparamètres
├── saved_model/             # Dossier pour sauvegarder les modèles
│   ├── point_decoder.pth
│   ├── image_encoder.pth
└── requirements.txt         # Dépendances du projet
```
