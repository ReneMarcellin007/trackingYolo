#!/bin/bash
# Script pour placer le modèle MultiDNN2 au bon endroit dans npu.yml

echo "🔧 Correction de la position du modèle MultiDNN2..."

# Supprimer l'entrée ajoutée à la fin
/home/jevois/jevois_docs/connect_jevois.sh cmd "sed -i '/# ========== YOLO avec IDs Aléatoires - Compatible MultiDNN2/,/^$/d' /jevoispro/share/dnn/npu.yml"

# Créer un fichier temporaire avec la bonne configuration
cat > /tmp/fix_multidnn2.sh << 'EOF'
#!/bin/bash
# Script exécuté sur le JeVois

# Sauvegarder
cp /jevoispro/share/dnn/npu.yml /jevoispro/share/dnn/npu.yml.backup.fix

# Trouver la ligne où insérer (juste après "# Object detection models")
LINE=$(grep -n "# Object detection models" /jevoispro/share/dnn/npu.yml | head -1 | cut -d: -f1)
LINE=$((LINE + 3))

# Créer la configuration à insérer
cat > /tmp/insert.txt << 'INNER_EOF'

# ========== PRIORITAIRE: YOLO avec IDs Aléatoires - MultiDNN2 Compatible ==========
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

INNER_EOF

# Insérer au bon endroit
head -n $((LINE-1)) /jevoispro/share/dnn/npu.yml > /tmp/npu_new.yml
cat /tmp/insert.txt >> /tmp/npu_new.yml
tail -n +$LINE /jevoispro/share/dnn/npu.yml >> /tmp/npu_new.yml

# Remplacer
mv /tmp/npu_new.yml /jevoispro/share/dnn/npu.yml

echo "✅ Position corrigée"
EOF

# Copier et exécuter le script de correction
/home/jevois/jevois_docs/connect_jevois.sh copy "/tmp/fix_multidnn2.sh" "/tmp/fix_multidnn2.sh"
/home/jevois/jevois_docs/connect_jevois.sh cmd "chmod +x /tmp/fix_multidnn2.sh && /tmp/fix_multidnn2.sh"

echo ""
echo "🔍 Vérification de la position..."
/home/jevois/jevois_docs/connect_jevois.sh cmd "sed -n '130,160p' /jevoispro/share/dnn/npu.yml | grep -A5 -B2 'AAA-yolov7'"

echo ""
echo "✅ Le modèle devrait maintenant apparaître en premier dans MultiDNN2!"