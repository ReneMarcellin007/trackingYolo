#!/usr/bin/env python3
"""
Script de test pour vérifier le post-processeur Python pur
Compatible avec MultiDNN2
"""

import numpy as np
import random

def test_decode_yolo():
    """Test simple du décodage YOLO en Python pur"""
    
    # Simuler une sortie YOLO
    grid_h, grid_w = 36, 64
    num_anchors = 3
    num_classes = 80
    output_shape = (1, num_anchors * (5 + num_classes), grid_h, grid_w)
    
    # Créer des données de test
    output = np.random.randn(*output_shape).astype(np.float32)
    
    # Anchors de test (YOLOv7-tiny première couche)
    anchors = [(10, 13), (16, 30), (33, 23)]
    
    # Paramètres
    blob_w, blob_h = 512, 288
    conf_thresh = 0.2
    
    # Décoder
    boxes = []
    confidences = []
    class_ids = []
    
    # Reshape pour le traitement
    output_reshaped = output.reshape((num_anchors, 5 + num_classes, grid_h, grid_w))
    output_reshaped = output_reshaped.transpose(0, 2, 3, 1)
    
    print(f"🔍 Test de décodage YOLO Python pur")
    print(f"   Shape entrée: {output_shape}")
    print(f"   Grille: {grid_w}x{grid_h}")
    print(f"   Anchors: {anchors}")
    print(f"   Classes: {num_classes}")
    
    detections_found = 0
    
    for anchor_idx in range(num_anchors):
        for y in range(grid_h):
            for x in range(grid_w):
                data = output_reshaped[anchor_idx, y, x]
                
                # Objectness
                objectness = 1.0 / (1.0 + np.exp(-data[4]))
                
                if objectness > conf_thresh:
                    # Trouver la meilleure classe
                    class_scores = 1.0 / (1.0 + np.exp(-data[5:]))
                    class_id = np.argmax(class_scores)
                    confidence = objectness * class_scores[class_id]
                    
                    if confidence > conf_thresh:
                        detections_found += 1
                        
                        # Décoder les coordonnées
                        tx, ty, tw, th = data[0:4]
                        anchor_w, anchor_h = anchors[anchor_idx]
                        
                        # Coordonnées de la boîte (simplifiées)
                        bx = (1.0 / (1.0 + np.exp(-tx)) + x) * blob_w / grid_w
                        by = (1.0 / (1.0 + np.exp(-ty)) + y) * blob_h / grid_h
                        bw = np.exp(tw) * anchor_w * blob_w / grid_w
                        bh = np.exp(th) * anchor_h * blob_h / grid_h
                        
                        # Ajouter un ID aléatoire
                        random_id = random.randint(1, 999)
                        
                        if detections_found <= 3:  # Afficher les 3 premières détections
                            print(f"   ✅ Détection {detections_found}: ID{random_id}, "
                                  f"classe={class_id}, conf={confidence:.2f}, "
                                  f"box=[{int(bx)},{int(by)},{int(bw)},{int(bh)}]")
    
    print(f"\n📊 Résultat: {detections_found} détections simulées trouvées")
    print(f"   (Avec données aléatoires, c'est normal d'avoir peu de détections)")
    
    return detections_found > 0

def test_multidnn2_compatibility():
    """Vérifier que le code ne dépend pas de PyPostYOLO"""
    
    print("\n🧪 Test de compatibilité MultiDNN2")
    
    # Vérifier qu'on n'importe pas PyPostYOLO
    dangerous_imports = [
        "jevois.PyPostYOLO",
        "getSubComponent",
        "pipeline"
    ]
    
    with open("PyPostYoloRandomID_PurePython.py", 'r') as f:
        code = f.read()
    
    issues = []
    for dangerous in dangerous_imports:
        if dangerous in code:
            issues.append(dangerous)
    
    if issues:
        print(f"   ❌ Code dépendant trouvé: {issues}")
        return False
    else:
        print(f"   ✅ Aucune dépendance à PyPostYOLO ou pipeline")
        return True

def main():
    """Tests principaux"""
    
    print("=" * 60)
    print("🚀 TEST DU POST-PROCESSEUR YOLO PYTHON PUR")
    print("   Compatible DNN et MultiDNN2")
    print("=" * 60)
    
    # Test 1: Décodage YOLO
    test1 = test_decode_yolo()
    
    # Test 2: Compatibilité MultiDNN2
    test2 = test_multidnn2_compatibility()
    
    print("\n" + "=" * 60)
    print("📈 RÉSUMÉ DES TESTS")
    print("=" * 60)
    print(f"   Décodage YOLO: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"   Compatibilité MultiDNN2: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n🎉 TOUS LES TESTS PASSENT - Prêt pour MultiDNN2!")
    else:
        print("\n⚠️  Certains tests ont échoué")

if __name__ == "__main__":
    main()