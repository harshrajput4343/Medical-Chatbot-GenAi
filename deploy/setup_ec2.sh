#!/bin/bash
# Complete EC2 Ubuntu 22.04 setup for Medical Chatbot v2

set -e

echo "Starting Medical Chatbot v2 Setup..."

# 1. Update and Upgrade
sudo apt update && sudo apt upgrade -y

# 2. Install Dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip git nginx certbot python3-certbot-nginx curl

# 3. Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# 4. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 5. Application Directory
sudo mkdir -p /opt/medbot
sudo chown $USER:$USER /opt/medbot

# 6. Setup Firewall
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable

echo "System ready. Please clone your repo into /opt/medbot and configure .env"
