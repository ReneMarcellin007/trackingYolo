#!/usr/bin/env python3
"""
Corrige tous les modèles pour qu'ils fonctionnent comme yolov7-hires-tracking-filtered
"""

import os
import glob

# Liste des fichiers à corriger
models = [
    '/jevoispro/share/pydnn/post/PyPostYolov7ColorTracker.py',
    '/jevoispro/share/pydnn/post/PyPostYolov7MultiDNN.py',
    '/jevoispro/share/pydnn/post/PyPostYolov7MultiDNNFiltered.py',
    '/jevoispro/share/pydnn/post/PyPostYolov7Simple.py'
]

for model_file in models:
    if not os.path.exists(model_file):
        print(f"⚠️  {model_file} n'existe pas")
        continue
        
    print(f"Correction de {os.path.basename(model_file)}...")
    
    with open(model_file, 'r') as f:
        content = f.read()
    
    # Remplacements nécessaires
    if 'self.yolopp = jevois.PyPostYOLO()' in content:
        content = content.replace(
            'self.yolopp = jevois.PyPostYOLO()',
            'self.yolopp = None  # Will be initialized later'
        )
        print("  ✓ Corrigé l'initialisation de yolopp")
    
    # S'assurer que freeze vérifie si yolopp existe
    if 'self.yolopp.freeze(doit)' in content and 'if self.yolopp:' not in content:
        content = content.replace(
            'self.yolopp.freeze(doit)',
            'if self.yolopp: self.yolopp.freeze(doit)'
        )
        print("  ✓ Corrigé la méthode freeze")
    
    # S'assurer que l'initialisation se fait dans init() ou process()
    if 'def init(self):' in content and 'if self.yolopp is None:' not in content:
        # Ajouter l'initialisation dans init() si elle n'existe pas
        lines = content.split('\n')
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            if line.strip() == 'def init(self):':
                # Chercher la fin du docstring
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('"""'):
                    j += 1
                if j < len(lines) and '"""' in lines[j]:
                    # Insérer après le docstring
                    new_lines.append(lines[j])
                    new_lines.append('        # Initialize YOLO decoder if needed')
                    new_lines.append('        if self.yolopp is None:')
                    new_lines.append('            try:')
                    new_lines.append('                self.yolopp = jevois.PyPostYOLO()')
                    new_lines.append('            except:')
                    new_lines.append('                pass  # Will retry in process()')
                    i = j
        content = '\n'.join(new_lines)
        print("  ✓ Ajouté l'initialisation dans init()")
    
    # S'assurer que process() vérifie aussi
    if 'def process(self, outs, preproc):' in content:
        if 'if self.yolopp is None:' not in content.split('def process(self, outs, preproc):')[1].split('def ')[0]:
            # Ajouter vérification dans process
            lines = content.split('\n')
            new_lines = []
            for i, line in enumerate(lines):
                new_lines.append(line)
                if line.strip() == 'def process(self, outs, preproc):':
                    # Ajouter après la ligne de définition et docstring
                    j = i + 1
                    while j < len(lines) and (lines[j].strip().startswith('"""') or '"""' in lines[j]):
                        new_lines.append(lines[j])
                        j += 1
                    new_lines.append('        # Ensure YOLO decoder is initialized')
                    new_lines.append('        if self.yolopp is None:')
                    new_lines.append('            try:')
                    new_lines.append('                self.yolopp = jevois.PyPostYOLO()')
                    new_lines.append('            except:')
                    new_lines.append('                return  # Cannot process without decoder')
                    i = j - 1
            content = '\n'.join(new_lines)
            print("  ✓ Ajouté vérification dans process()")
    
    # Sauvegarder
    with open(model_file, 'w') as f:
        f.write(content)
    
    print(f"✅ {os.path.basename(model_file)} corrigé!\n")

print("🎉 Tous les modèles ont été corrigés!")
print("\nLes modèles suivants devraient maintenant fonctionner:")
print("- yolov7-hires-tracking")
print("- yolov7-multidnn") 
print("- yolov7-multidnn-vehicles")
print("- yolov7-simple")