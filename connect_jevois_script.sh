#!/bin/bash

# Configuration JeVois Pro
JEVOIS_IP="192.168.0.100"
JEVOIS_USER="root"
JEVOIS_PASSWORD="jevois"  # Remplacez par votre mot de passe

# Fonction de connexion
connect_jevois() {
    echo "Connexion au JeVois Pro..."
    sshpass -p "$JEVOIS_PASSWORD" ssh -o StrictHostKeyChecking=no "$JEVOIS_USER@$JEVOIS_IP"
}

# Fonction pour exécuter une commande à distance
run_command() {
    if [ -z "$1" ]; then
        echo "Usage: run_command 'votre_commande'"
        return 1
    fi
    echo "Exécution de: $1"
    sshpass -p "$JEVOIS_PASSWORD" ssh -o StrictHostKeyChecking=no "$JEVOIS_USER@$JEVOIS_IP" "$1"
}

# Fonction pour copier des fichiers
copy_file() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: copy_file fichier_local chemin_distant"
        return 1
    fi
    echo "Copie de $1 vers $2"
    sshpass -p "$JEVOIS_PASSWORD" scp -o StrictHostKeyChecking=no "$1" "$JEVOIS_USER@$JEVOIS_IP:$2"
}

# Menu principal
case "$1" in
    "connect")
        connect_jevois
        ;;
    "cmd")
        run_command "$2"
        ;;
    "copy")
        copy_file "$2" "$3"
        ;;
    *)
        echo "Usage: $0 {connect|cmd|copy}"
        echo "  connect          - Se connecter au JeVois Pro"
        echo "  cmd 'commande'   - Exécuter une commande"
        echo "  copy src dst     - Copier un fichier"
        ;;
esac