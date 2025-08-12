#!/bin/bash
# Script d'installation de la solution MultiDNN2 pour YOLO avec IDs aléatoires

echo "================================================"
echo "🚀 INSTALLATION SOLUTION MULTIDNN2"
echo "   YOLO avec IDs Aléatoires - Python Pur"
echo "================================================"

# Vérifier qu'on est dans le bon répertoire
if [ ! -f "PyPostYoloRandomID_PurePython.py" ]; then
    echo "❌ Erreur: PyPostYoloRandomID_PurePython.py non trouvé"
    echo "   Exécutez ce script depuis /home/jevois/jevois_docs/"
    exit 1
fi

echo ""
echo "📦 Copie du post-processeur Python pur..."
/home/jevois/jevois_docs/connect_jevois.sh copy \
    "/home/jevois/jevois_docs/PyPostYoloRandomID_PurePython.py" \
    "/jevoispro/share/pydnn/post/PyPostYoloRandomID_PurePython.py"

echo ""
echo "✅ Vérification de la copie..."
/home/jevois/jevois_docs/connect_jevois.sh cmd \
    "ls -la /jevoispro/share/pydnn/post/PyPostYoloRandomID_PurePython.py"

echo ""
echo "📝 Ajout de la configuration dans npu.yml..."

# Créer le fichier de configuration temporaire
cat > /tmp/multidnn2_config.txt << 'EOF'

# ========== YOLO avec IDs Aléatoires - Compatible MultiDNN2 ==========
AAA-yolov7-randomid-multidnn2:
  comment: "YOLOv7-tiny avec IDs aléatoires - Compatible MultiDNN2"
  url: "MULTIDNN2_COMPATIBLE"
  model: "npu/detection/yolov7-tiny-512x288.nb"
  postproc: Python
  pypost: "pydnn/post/PyPostYoloRandomID_PurePython.py"
  detecttype: RAWYOLO
  intensors: "NCHW:8U:1x3x288x512:AA:0.003921568393707275:0"
  outtensors: "NCHW:8U:1x255x36x64:AA:0.003916095942258835:0, NCHW:8U:1x255x18x32:AA:0.00392133416607976:0, NCHW:8U:1x255x9x16:AA:0.003921062219887972:0"
  anchors: "10,13, 16,30, 33,23;   30,61, 62,45, 59,119;   116,90, 156,198, 373,326"
  scalexy: 2.0
  sigmoid: false
  classes: "dnn/labels/coco-labels.txt"
  cthresh: 20
  nms: 45

EOF

# Copier la configuration
/home/jevois/jevois_docs/connect_jevois.sh copy \
    "/tmp/multidnn2_config.txt" \
    "/tmp/multidnn2_config.txt"

# Sauvegarder l'ancien npu.yml
echo ""
echo "💾 Sauvegarde de npu.yml..."
/home/jevois/jevois_docs/connect_jevois.sh cmd \
    "cp /jevoispro/share/dnn/npu.yml /jevoispro/share/dnn/npu.yml.backup.multidnn2"

# Ajouter la configuration
echo ""
echo "➕ Ajout de la configuration..."
/home/jevois/jevois_docs/connect_jevois.sh cmd \
    "cat /tmp/multidnn2_config.txt >> /jevoispro/share/dnn/npu.yml"

# Vérifier l'ajout
echo ""
echo "🔍 Vérification de l'ajout..."
/home/jevois/jevois_docs/connect_jevois.sh cmd \
    "grep -A5 'AAA-yolov7-randomid-multidnn2' /jevoispro/share/dnn/npu.yml"

echo ""
echo "🐍 Test de compilation Python..."
/home/jevois/jevois_docs/connect_jevois.sh cmd \
    "python3 -m py_compile /jevoispro/share/pydnn/post/PyPostYoloRandomID_PurePython.py"

echo ""
echo "================================================"
echo "✅ INSTALLATION TERMINÉE!"
echo "================================================"
echo ""
echo "📋 UTILISATION:"
echo "   1. Ouvrez l'interface JeVois-Pro"
echo "   2. Sélectionnez le module MultiDNN2"
echo "   3. Dans Pipeline 0, choisissez:"
echo "      NPU:Detect:AAA-yolov7-randomid-multidnn2"
echo ""
echo "🎯 AVANTAGES:"
echo "   ✅ Compatible DNN et MultiDNN2"
echo "   ✅ Pas de dépendance PyPostYOLO"
echo "   ✅ IDs aléatoires sur chaque frame"
echo "   ✅ Apparaît en premier (AAA-)"
echo ""
echo "📚 Documentation: SOLUTIONS_MULTIDNN2.md"
echo "================================================"