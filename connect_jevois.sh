#!/bin/bash

# Configuration JeVois Pro
JEVOIS_IP="192.168.0.100"
JEVOIS_USER="root"
JEVOIS_PASSWORD="jevois"

case "$1" in
    "cmd")
        sshpass -p "$JEVOIS_PASSWORD" ssh -o StrictHostKeyChecking=no "$JEVOIS_USER@$JEVOIS_IP" "$2"
        ;;
    "copy")
        sshpass -p "$JEVOIS_PASSWORD" scp -o StrictHostKeyChecking=no "$2" "$JEVOIS_USER@$JEVOIS_IP:$3"
        ;;
    *)
        sshpass -p "$JEVOIS_PASSWORD" ssh -o StrictHostKeyChecking=no "$JEVOIS_USER@$JEVOIS_IP"
        ;;
esac