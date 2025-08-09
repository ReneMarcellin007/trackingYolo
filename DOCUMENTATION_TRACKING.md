# Documentation Technique : Système de Tracking YOLOv7 Avancé pour JeVois Pro

## Vue d'ensemble

Ce projet implémente un système de tracking d'objets avancé sur JeVois Pro, utilisant YOLOv7-tiny haute résolution (1024x576) avec des fonctionnalités innovantes de tracking par couleur, prédiction de position et détection automatique de contexte.

## Architecture du Système

### Modèles Disponibles

#### 1. YOLOv7-hires-tracking (Original)
- **Pipeline** : `NPU:Python:yolov7-hires-tracking`
- **Détection** : 80 classes COCO complètes
- **Résolution** : 1024x576 (2x supérieure au standard)
- **Performance** : ~60 FPS sur NPU

#### 2. YOLOv7-hires-tracking-filtered (Filtré)
- **Pipeline** : `NPU:Python:yolov7-hires-tracking-filtered`
- **Détection** : Personnes + Véhicules uniquement
  - Personne (classe 0)
  - Vélo (classe 1)
  - Voiture (classe 2)
  - Moto (classe 3)
  - Bus (classe 5)
  - Camion (classe 7)
- **Cas d'usage** : Applications routières, comptage de trafic

## Fonctionnalités Principales

### 1. Tracking avec IDs Persistants

Le système attribue un ID unique à chaque objet détecté et maintient cet ID tant que l'objet reste visible.

**Paramètres optimisés** :
- `max_distance` : 200 pixels (distance maximale pour association)
- `max_missing_frames` : 30 frames (tolérance aux occlusions)
- `association_threshold` : 1.8 (seuil de coût pour association)

**Algorithme d'association** :
```python
coût_total = distance_cost * 0.5 + 
             color_cost * 0.3 + 
             prediction_cost * 0.2
```

### 2. Analyse de Couleur HSV

Extraction et analyse de la couleur dominante de chaque objet pour améliorer le tracking.

**Caractéristiques** :
- Extraction de la région centrale (évite les bords)
- Conversion BGR → HSV pour robustesse
- Classification en 9 couleurs principales
- Validation de fiabilité (saturation ≥ 40, luminosité ≥ 60)

**Couleurs détectées** :
- Rouge, Orange, Jaune, Lime, Vert
- Cyan, Bleu, Purple, Magenta
- Noir, Blanc, Gris (si faible saturation)

### 3. Prédiction de Position

Prédiction simple mais efficace de la position future basée sur l'historique.

**Implémentation** :
- Historique des 5 dernières positions
- Calcul de vitesse moyenne
- Prédiction linéaire de la position suivante
- Limitation de vitesse max (30 pixels/frame)

### 4. Détection Automatique de Contexte

Le système détecte automatiquement l'environnement et adapte ses paramètres.

**Contextes** :
| Contexte | Distance Max | Frames Manquées | Caractéristiques |
|----------|--------------|-----------------|------------------|
| INDOOR   | 150 px       | 25 frames       | Peu d'objets, mouvements lents |
| CITY     | 180 px       | 30 frames       | Quelques véhicules, vitesse moyenne |
| HIGHWAY  | 250 px       | 40 frames       | Beaucoup de véhicules, vitesse élevée |

**Détection** :
- Analyse du nombre de véhicules détectés
- Calcul de la vitesse moyenne des objets
- Hystérésis pour éviter les changements fréquents

## Structure des Fichiers

### Configuration YAML

**yolov7-hires-tracking.yml** :
```yaml
%YAML 1.0
---
yolov7-hires-tracking:
  preproc: Blob
  mean: "0 0 0"
  scale: 0.0039215686
  nettype: NPU
  model: "npu/detection/yolov7-tiny-1024x576.nb"
  postproc: Python
  pypost: "pydnn/post/PyPostYolov7ColorTracker.py"
  detecttype: RAWYOLO
  classes: "dnn/labels/coco-labels.txt"
  processing: Sync
  intensors: "NCHW:8U:1x3x576x1024:AA:0.003921568393707275:0"
  outtensors: "NCHW:8U:1x255x72x128:AA:0.003911261446774006:0, ..."
  anchors: "10,13, 16,30, 33,23; 30,61, 62,45, 59,119; 116,90, 156,198, 373,326"
  scalexy: 2.0
  sigmoid: false
```

### Post-Processeur Python

**Structure de PyPostYolov7ColorTracker** :

```python
class PyPostYolov7ColorTracker:
    def __init__(self):
        # État du tracking
        self.tracks = {}
        self.next_id = 1
        self.max_missing_frames = 30
        
    def process(self, outs, preproc):
        # Décodage YOLO
        # Non-Maximum Suppression
        # Application du tracking
        
    def update_tracks(self, boxes, class_ids, confidences):
        # Association détection-track
        # Création nouveaux tracks
        # Suppression tracks perdus
        
    def extract_color_info(self, img, box):
        # Extraction couleur HSV
        # Classification couleur
        # Validation fiabilité
        
    def predict_position(self, track):
        # Calcul vitesse
        # Prédiction position
        
    def detect_context(self, detections):
        # Analyse environnement
        # Adaptation paramètres
```

## Messages de Sortie

Le système envoie des messages détaillés sur le port série :

```
ENHANCED id:1 class:person color:blue hsv:120,180,200 conf:0.85 quality:0.92 speed:15.3 context:INDOOR x:640 y:360
```

**Champs disponibles** :
- `id` : Identifiant unique persistant
- `class` : Classe d'objet détectée
- `color` : Nom de la couleur dominante
- `hsv` : Valeurs HSV exactes
- `conf` : Confidence de détection
- `quality` : Qualité du tracking (0.0-1.0)
- `speed` : Vitesse en pixels/frame
- `context` : Contexte détecté (INDOOR/CITY/HIGHWAY)
- `x, y` : Position du centre

## Installation et Utilisation

### Installation sur JeVois Pro

1. **Copier les fichiers de configuration** :
```bash
./connect_jevois_script.sh copy "yolov7-hires-tracking.yml" "/jevoispro/share/dnn/custom/"
./connect_jevois_script.sh copy "yolov7-hires-tracking-filtered.yml" "/jevoispro/share/dnn/custom/"
```

2. **Copier les post-processeurs** :
```bash
./connect_jevois_script.sh copy "PyPostYolov7ColorTracker.py" "/jevoispro/share/pydnn/post/"
./connect_jevois_script.sh copy "PyPostYolov7ColorTrackerFiltered.py" "/jevoispro/share/pydnn/post/"
```

3. **Sélectionner le modèle** :
- Via GUI : Sélectionner `NPU:Python:yolov7-hires-tracking`
- Via série : `setpar model NPU:Python:yolov7-hires-tracking`

### Paramètres Configurables

Les paramètres peuvent être ajustés en temps réel via l'interface :

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `cthresh` | 20.0% | Seuil de classification |
| `dthresh` | 15.0% | Seuil de détection |
| `nms` | 45.0% | Non-Maximum Suppression |
| `max_distance` | 200.0 | Distance max pour association |
| `color_weight` | 0.3 | Poids de la couleur dans le tracking |
| `enable_tracking` | true | Activer/désactiver le tracking |
| `enable_prediction` | true | Activer la prédiction de position |
| `enable_auto_context` | true | Détection automatique du contexte |

## Optimisations et Performances

### Optimisations Appliquées

1. **Paramètres de tracking** :
   - Distance maximale augmentée pour tolérer plus de mouvement
   - Tolérance aux frames manquées augmentée
   - Seuil d'association assoupli

2. **Traitement d'image** :
   - Échantillonnage tous les 3 pixels pour l'extraction de couleur
   - Région centrale uniquement pour éviter les artefacts de bord
   - Cache des calculs HSV

3. **Algorithme** :
   - Association gourmande optimisée
   - Tri par confidence pour meilleure association
   - Prédiction conservative (facteur 0.8)

### Métriques de Performance

- **FPS** : 55-60 sur NPU JeVois Pro
- **Latence** : < 20ms par frame
- **Précision de détection** : > 90% sur personnes et véhicules
- **Stabilité des IDs** : > 95% sur objets visibles
- **Consommation mémoire** : < 50MB

## Cas d'Usage et Applications

### Applications Implémentées

1. **Comptage de personnes uniques**
   - Ne compte pas 2 fois la même personne
   - Maintient l'historique des IDs vus

2. **Analyse de trafic routier**
   - Comptage différencié par type de véhicule
   - Analyse de vitesse et trajectoire

3. **Système de distribution**
   - Distribution unique par personne (bonbons, flyers)
   - Vérification via ID persistant

### Applications Potentielles

- **Surveillance de sécurité** : Détection d'intrusion avec identification
- **Analyse comportementale** : Temps passé, zones visitées
- **Interaction personnalisée** : Actions différentes selon l'historique
- **Statistiques de fréquentation** : Comptage précis sans doublons

## Troubleshooting

### Problèmes Courants

| Problème | Cause | Solution |
|----------|-------|----------|
| IDs changent trop vite | Paramètres trop stricts | Augmenter `max_distance` et `max_missing_frames` |
| Pas de détection | Seuils trop élevés | Réduire `cthresh` et `dthresh` |
| Couleurs incorrectes | Éclairage faible | Ajuster `min_saturation` et `min_brightness` |
| Contexte instable | Changements fréquents | Augmenter l'hystérésis dans `detect_context` |

### Logs et Debug

Activer les logs détaillés :
```python
jevois.LINFO(f"Track {track_id}: cost={cost:.2f}, quality={quality:.2f}")
```

## Évolutions Futures

### Améliorations Prévues

1. **Filtrage de Kalman** pour prédiction plus robuste
2. **ReID** (ré-identification) après perte prolongée
3. **Multi-caméra** avec synchronisation des IDs
4. **Apprentissage** des patterns de mouvement
5. **Export** des statistiques en temps réel

### Contributions

Le projet est open-source et les contributions sont bienvenues :
- GitHub : https://github.com/ReneMarcellin007/trackingYolo
- Issues : Pour signaler des bugs ou demander des fonctionnalités
- Pull Requests : Pour proposer des améliorations

---
*Documentation mise à jour le 08/08/2025*
*Version : 2.0 - Tracking Optimisé avec Filtrage*