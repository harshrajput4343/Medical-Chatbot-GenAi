# MedBot AI — AWS EC2 Deployment Guide

## Prerequisites
- EC2 instance running with Docker + Docker Compose
- Nginx reverse proxy configured (see `nginx.conf`)
- SSL via Certbot / DuckDNS (`harshmedbot.duckdns.org`)
- `.env` file configured on server at `/opt/medbot/.env`

---

## 1. Local — Push Changes

```bash
# Check what's changed
git status

# Stage, commit, push
git add .
git commit -m "feat: landing page + navbar auth buttons"
git push origin main
```

---

## 2. EC2 — Pull & Rebuild (Docker)

```bash
# SSH into your instance
ssh -i ~/.ssh/your-key.pem ubuntu@<EC2-PUBLIC-IP>

# Navigate to project
cd /opt/medbot

# Pull latest code
git pull origin main

# Rebuild and restart containers (zero-downtime with --no-cache)
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Verify containers are running
docker-compose ps
docker-compose logs --tail=50 web
```

### Alternative: Direct uvicorn/gunicorn (no Docker)

```bash
cd /opt/medbot
git pull origin main
pip install -r requirements.txt   # only if new deps added
sudo systemctl restart medbot     # restart the systemd service
sudo systemctl status medbot      # verify it's running
```

---

## 3. Verify SSL & Routes

```bash
# Landing page
curl -I https://harshmedbot.duckdns.org/
# Expected: HTTP/2 200

# Login page
curl -I https://harshmedbot.duckdns.org/auth/login
# Expected: HTTP/2 200

# Register page
curl -I https://harshmedbot.duckdns.org/auth/register
# Expected: HTTP/2 200

# Health check
curl https://harshmedbot.duckdns.org/api/v1/health
# Expected: JSON with status "healthy"

# Check DuckDNS renewal cron is active
crontab -l | grep duckdns
```

---

## 4. Route Safety Checklist

| Route | Method | Expected |
|---|---|---|
| `/` | GET | New landing page (unauthenticated) / Redirect to `/dashboard` (authenticated) |
| `/auth/login` | GET | Login form with navbar |
| `/auth/register` | GET | Register form with navbar |
| `/auth/login` | POST | Auth → redirect to `/dashboard` |
| `/auth/register` | POST | Create user → redirect to `/dashboard` |
| `/dashboard` | GET | Dashboard (auth required) |
| `/chat` | GET | Chat interface (auth required) |
| `/api/v1/health` | GET | Health check JSON |

---

## 5. Rollback (if needed)

```bash
# On EC2
cd /opt/medbot
git log --oneline -5          # find the previous commit hash
git checkout <COMMIT_HASH>    # rollback
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## Files Changed in This Release

| File | Change |
|---|---|
| `app/routers/dashboard.py` | `GET /` renders `landing.html` instead of redirecting |
| `app/templates/landing.html` | **NEW** — Landing page with glassmorphism design |
| `app/templates/auth/login.html` | Added thin navbar above form |
| `app/templates/auth/register.html` | Added thin navbar above form |
| `app/templates/base.html` | Added favicon + OG meta tags |
| `static/css/landing.css` | **NEW** — Landing page animations & glass styles |
| `deploy/DEPLOY.md` | **NEW** — This file |
