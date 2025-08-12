#!/bin/bash

# Installation automatique du module YOLOv7 avec IDs aléatoires pour JeVois-Pro
# Usage: ./install.sh

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Installation Module YOLOv7 Random ID pour JeVois-Pro   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Vérifier qu'on est sur JeVois-Pro
if [ ! -d "/jevoispro/share/dnn" ]; then
    echo "❌ Erreur : Ce script doit être exécuté sur JeVois-Pro"
    echo "   Répertoire /jevoispro/share/dnn non trouvé"
    exit 1
fi

# Backup des fichiers existants
echo "📦 Sauvegarde des fichiers existants..."
if [ -f "/jevoispro/share/dnn/npu.yml" ]; then
    cp /jevoispro/share/dnn/npu.yml /jevoispro/share/dnn/npu.yml.backup.$(date +%Y%m%d_%H%M%S)
    echo "   ✅ npu.yml sauvegardé"
fi

# Copier le post-processeur Python
echo ""
echo "📝 Installation du post-processeur Python..."
cp PyPostYoloRandomID.py /jevoispro/share/pydnn/post/
chmod 755 /jevoispro/share/pydnn/post/PyPostYoloRandomID.py
echo "   ✅ PyPostYoloRandomID.py installé"

# Vérifier la syntaxe Python
echo ""
echo "🔍 Vérification de la syntaxe Python..."
if python3 -m py_compile /jevoispro/share/pydnn/post/PyPostYoloRandomID.py 2>/dev/null; then
    echo "   ✅ Syntaxe Python correcte"
else
    echo "   ❌ Erreur de syntaxe Python !"
    exit 1
fi

# Ajouter l'entrée dans npu.yml
echo ""
echo "⚙️  Configuration dans npu.yml..."

# Vérifier si l'entrée existe déjà
if grep -q "AAA-yolov7-randomid:" /jevoispro/share/dnn/npu.yml; then
    echo "   ⚠️  L'entrée AAA-yolov7-randomid existe déjà"
    echo "   Suppression de l'ancienne entrée..."
    sed -i '/AAA-yolov7-randomid:/,/^[[:space:]]*$/d' /jevoispro/share/dnn/npu.yml
fi

# Créer l'entrée de configuration
cat > /tmp/randomid_config.yml << 'EOF'

####################################################################################################
# Object detection models - MODÈLE PRIORITAIRE
####################################################################################################

# -------------------- YOLOv7 avec IDs aléatoires - EN PREMIER
AAA-yolov7-randomid:
  comment: "YOLOv7-tiny avec IDs aléatoires - APPARAIT EN PREMIER"
  model: "npu/detection/yolov7-tiny-512x288.nb"
  postproc: Python
  pypost: "pydnn/post/PyPostYoloRandomID.py"
  detecttype: RAWYOLO
  intensors: "NCHW:8U:1x3x288x512:AA:0.003921568393707275:0"
  outtensors: "NCHW:8U:1x255x36x64:AA:0.003916095942258835:0, NCHW:8U:1x255x18x32:AA:0.00392133416607976:0, NCHW:8U:1x255x9x16:AA:0.003921062219887972:0"
  anchors: "10,13, 16,30, 33,23;   30,61, 62,45, 59,119;   116,90, 156,198, 373,326"
  scalexy: 2.0
  sigmoid: false
  classes: "dnn/labels/coco-labels.txt"

EOF

# Trouver où insérer (avant la section Object detection)
LINE=$(grep -n "# Object detection models" /jevoispro/share/dnn/npu.yml | head -1 | cut -d: -f1)

if [ -z "$LINE" ]; then
    # Si on ne trouve pas, chercher avant le premier yolo
    LINE=$(grep -n "^yolo.*:" /jevoispro/share/dnn/npu.yml | head -1 | cut -d: -f1)
fi

if [ ! -z "$LINE" ]; then
    # Insérer notre configuration
    head -n $((LINE-1)) /jevoispro/share/dnn/npu.yml > /tmp/npu_new.yml
    cat /tmp/randomid_config.yml >> /tmp/npu_new.yml
    tail -n +$LINE /jevoispro/share/dnn/npu.yml >> /tmp/npu_new.yml
    mv /tmp/npu_new.yml /jevoispro/share/dnn/npu.yml
    echo "   ✅ Configuration ajoutée dans npu.yml"
else
    echo "   ❌ Impossible d'ajouter la configuration"
    exit 1
fi

# Vérifier l'installation
echo ""
echo "🔍 Vérification de l'installation..."
if grep -q "AAA-yolov7-randomid:" /jevoispro/share/dnn/npu.yml; then
    echo "   ✅ Module trouvé dans npu.yml"
else
    echo "   ❌ Module non trouvé dans npu.yml"
    exit 1
fi

if [ -f "/jevoispro/share/pydnn/post/PyPostYoloRandomID.py" ]; then
    echo "   ✅ Post-processeur Python présent"
else
    echo "   ❌ Post-processeur Python manquant"
    exit 1
fi

# Nettoyage
rm -f /tmp/randomid_config.yml 2>/dev/null

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║            ✅ INSTALLATION TERMINÉE AVEC SUCCÈS          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "📍 Pour utiliser le module :"
echo "   1. Démarrez le module DNN sur JeVois-Pro"
echo "   2. Dans le paramètre 'pipe', sélectionnez :"
echo "      NPU:Python:AAA-yolov7-randomid"
echo ""
echo "📊 Format d'affichage : ID[numéro]:[classe]: [confiance]%"
echo "   Exemple : ID42:person: 95.3%"
echo ""
echo "💡 Note : Les IDs changent à chaque frame (aléatoires)"
echo ""