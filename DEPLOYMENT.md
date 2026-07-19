# Deploying `ai-service` as a Standalone Server (Production, no Docker)

This guide sets up the AI digitization service directly on a Linux server (e.g.
`py.orthosage.co`) using a Python virtual environment, systemd, and Nginx — no
Docker involved.

For local development via the Docker Compose stack, see the root `docker-compose.yml`
instead; this guide is specifically for a bare server.

---

## 0. Prerequisites on the server

- A Linux server (Ubuntu/Debian assumed below) with DNS for `py.orthosage.co`
  already pointing at its public IP.
- Ports `80` and `443` open in the firewall/security group. Port `8000` (the app's
  internal port) should **not** be open to the public — see [Security](#security).

Install system packages:
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0 \
    nginx certbot python3-certbot-nginx git
```
(`libgl1`/`libglib2.0-0` are required by Pillow/torchvision's image codecs.)

---

## 1. Get the code onto the server

```bash
sudo mkdir -p /opt/orthosage
sudo chown $USER:$USER /opt/orthosage
git clone <your-repo-url> /opt/orthosage/webceph
cd /opt/orthosage/webceph/ai-service
```

(Later deploys: `cd /opt/orthosage/webceph && git pull`.)

---

## 2. Create the virtual environment and install dependencies

From `ai-service/`:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# CPU-only torch/torchvision build — much smaller and faster than the default GPU wheel
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
deactivate
```

Quick sanity check it's installed correctly:
```bash
./venv/bin/python -c "import torch, torchvision, fastapi; print('ok')"
```

---

## 3. (Optional) Trained model weights

If you have a trained `.pth` weights file, copy it to the server, e.g.:
```bash
sudo mkdir -p /opt/orthosage/weights
sudo cp /path/to/model.pth /opt/orthosage/weights/model.pth
```
Without this, the service runs with randomly initialized weights — it responds
correctly and always returns in-bounds landmark coordinates, but the positions
won't be meaningful predictions until real weights are supplied.

---

## 4. Run it as a systemd service

Create `/etc/systemd/system/orthosage-ai.service`:
```ini
[Unit]
Description=Orthosage AI digitization service
After=network.target

[Service]
WorkingDirectory=/opt/orthosage/webceph/ai-service
Environment=MODEL_WEIGHTS_PATH=/opt/orthosage/weights/model.pth
ExecStart=/opt/orthosage/webceph/ai-service/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3
User=www-data

[Install]
WantedBy=multi-user.target
```

Notes:
- `--host 127.0.0.1` binds it to localhost only — Nginx reaches it locally, the
  internet cannot reach port 8000 directly. This is the important part for
  [security](#security).
- Drop the `Environment=MODEL_WEIGHTS_PATH=...` line if you skipped step 3.
- `User=www-data` needs read access to the app directory and the weights file:
  ```bash
  sudo chown -R www-data:www-data /opt/orthosage/webceph/ai-service /opt/orthosage/weights
  ```

Enable and start it:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now orthosage-ai
sudo systemctl status orthosage-ai
```

Check logs:
```bash
sudo journalctl -u orthosage-ai -f
```
You should see uvicorn's startup log ending with `Application startup complete.`

---

## 5. Reverse proxy + TLS

Create an Nginx server block:

```nginx
# /etc/nginx/sites-available/py.orthosage.co
server {
    listen 80;
    server_name py.orthosage.co;

    client_max_body_size 20M; # radiograph uploads can be a few MB

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/py.orthosage.co /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Issue and auto-configure TLS (also adds the HTTPS server block + HTTP redirect)
sudo certbot --nginx -d py.orthosage.co
```

---

## 6. Verify

```bash
curl https://py.orthosage.co/health
# expect: {"status":"ok"}
```

If you get 403/404 instead of that JSON, the request isn't reaching uvicorn — check:
```bash
curl http://127.0.0.1:8000/health          # on the server itself — bypasses Nginx
sudo systemctl status orthosage-ai         # is the service actually running?
sudo nginx -t && sudo systemctl status nginx
```

Then test the real endpoint with a file upload:
```bash
curl -X POST https://py.orthosage.co/image-digitization \
  -F "image=@/path/to/sample-xray.jpg"
```

---

## 7. Point Laravel at it

On the **Laravel** app's live `.env` (a separate server from this one):
```
AI_SERVICE_URL=https://py.orthosage.co/image-digitization
```

---

## 8. Redeploying after code changes

```bash
cd /opt/orthosage/webceph
git pull
cd ai-service
source venv/bin/activate
pip install -r requirements.txt   # only needed if requirements.txt changed
deactivate
sudo systemctl restart orthosage-ai
sudo systemctl status orthosage-ai
```

---

## Security

This service has **no authentication, no CORS restriction, and no rate limiting**.
Anyone who can reach `/image-digitization` can submit images and consume CPU/memory.
Keep it locked down:
- uvicorn bound to `127.0.0.1` only (step 4), reached solely through Nginx on the
  same machine.
- Don't open port `8000` in any firewall/security-group rule — only `80`/`443`.
- Nginx is the only public entry point — consider an IP allowlist for Laravel's
  outbound IP if it's static, or a shared-secret header checked by Nginx before
  proxying, e.g.:
  ```nginx
  if ($http_x_internal_key != "some-long-random-value") {
      return 403;
  }
  ```
  with Laravel sending that header on every request to this service.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `curl https://py.orthosage.co/health` → 403 on `/`, 404 on `/health` | Nginx has no proxy config for this domain yet (server block missing or not enabled), or `orthosage-ai` service isn't running. |
| `curl http://127.0.0.1:8000/health` on the server fails to connect | The systemd service isn't running — `sudo systemctl status orthosage-ai`, then `sudo journalctl -u orthosage-ai -e` for the error. |
| 502 Bad Gateway from Nginx | uvicorn crashed or hasn't started yet — check `journalctl -u orthosage-ai -f`. |
| `ModuleNotFoundError` on startup | Dependencies weren't installed into the venv, or the systemd unit's `ExecStart` isn't pointing at `venv/bin/uvicorn`. |
| Permission denied reading weights file | `User=www-data` in the unit doesn't own/can't read the weights path — `chown` it as shown in step 4. |
| Requests hang or time out on large images | Increase `client_max_body_size` in the Nginx block; check server CPU load if inference itself is slow (no GPU by default). |
