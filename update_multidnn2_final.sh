#!/bin/bash
# Mise à jour finale pour MultiDNN2

echo "🔄 Mise à jour de la configuration pour utiliser le module MultiDNN2..."

# Script pour mettre à jour sur le JeVois
cat > /tmp/update_config.sh << 'EOF'
#!/bin/bash

# Remplacer le pypost dans la configuration existante
sed -i 's|pypost: "pydnn/post/PyPostYoloRandomID_PurePython.py"|pypost: "pydnn/post/PyPostYoloRandomID_MultiDNN2.py"|' /jevoispro/share/dnn/npu.yml

# Vérifier
grep -A5 "AAA-yolov7-randomid-multidnn2" /jevoispro/share/dnn/npu.yml
EOF

# Exécuter
/home/jevois/jevois_docs/connect_jevois.sh copy "/tmp/update_config.sh" "/tmp/update_config.sh"
/home/jevois/jevois_docs/connect_jevois.sh cmd "chmod +x /tmp/update_config.sh && /tmp/update_config.sh"

echo ""
echo "✅ Configuration mise à jour pour utiliser PyPostYoloRandomID_MultiDNN2"
echo ""
echo "📋 TESTEZ MAINTENANT :"
echo "   1. Redémarrez l'interface JeVois-Pro"
echo "   2. Sélectionnez MultiDNN2"  
echo "   3. Cherchez: AAA-yolov7-randomid-multidnn2"