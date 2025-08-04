# Configuration Claude - Projet YOLOv7 Tracking

## 🔗 Repository GitHub

**Lien** : https://github.com/ReneMarcellin007/trackingYolo.git

**Token d'accès** : `[TOKEN_PERSONNEL_GITHUB]` *(voir token privé)*

## 📡 Connexion JeVois Pro

### Utilisation du script de connexion :

```bash
./connect_jevois_script.sh
```

### Commandes disponibles :

**Se connecter en SSH** :
```bash
./connect_jevois_script.sh ssh
```

**Exécuter une commande** :
```bash
./connect_jevois_script.sh cmd "ls -la /jevoispro/share/pydnn/post/"
```

**Copier un fichier vers JeVois Pro** :
```bash
./connect_jevois_script.sh copy "fichier_local.py" "/jevoispro/share/pydnn/post/fichier_distant.py"
```

### 🔐 Authentification
- **IP JeVois Pro** : `192.168.1.100` (par défaut)
- **Utilisateur** : `jevois`
- **Mot de passe** : `jevois`

## 📝 Notes importantes

- Le token GitHub est personnel et doit rester confidentiel
- Le script `connect_jevois_script.sh` doit être exécutable (`chmod +x`)
- Pour modifier l'IP du JeVois Pro, éditer la variable dans le script
- Les fichiers Python doivent être copiés dans `/jevoispro/share/pydnn/post/`

## 🎯 Fichiers du projet

- `PyPostYolov7ColorTracker_Final.py` - Tracker enhanced
- `DOCUMENTATION_TRACKING_COULEUR.md` - Documentation technique
- `connect_jevois_script.sh` - Script de connexion
- `README.md` - Documentation principale
- `claude.md` - Ce fichier de configuration

---
*Généré par Claude pour le projet de tracking YOLOv7 avec prédiction et détection de contexte*