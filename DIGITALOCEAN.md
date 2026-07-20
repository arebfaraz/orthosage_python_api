# Hosting `ai-service` on a DigitalOcean Droplet

This covers the DigitalOcean-specific setup (Droplet, firewall, DNS). Once the
Droplet is reachable over SSH, everything from "install system packages" onward
is identical to [`DEPLOYMENT.md`](./DEPLOYMENT.md) — this doc hands off to that
one rather than repeating it.

---

## 1. Create the Droplet

- **Image**: Ubuntu 24.04 LTS (matches the `apt-get` commands in `DEPLOYMENT.md`).
- **Size**: this service does CPU-only PyTorch inference on a single 768×768
  image per request — not GPU/heavy-compute, but torch + torchvision + the
  loaded model want headroom. Minimum workable size is the **Basic, Regular,
  2 GB RAM / 1 vCPU** plan. If you expect more than light/occasional traffic,
  go **4 GB / 2 vCPU** instead — uvicorn's default single worker processes one
  request at a time, so concurrent requests queue rather than parallelize
  unless you add more, and more vCPUs help there.
- **Authentication**: SSH key (not password).
- **Region**: pick whichever is closest to your Laravel server, to keep the
  server-to-server request latency low.
- **Hostname**: e.g. `ai-orthosage` — cosmetic, doesn't need to match the domain.

## 2. DigitalOcean Cloud Firewall

Create a Cloud Firewall (Networking → Firewalls) and attach it to the Droplet:

| Type | Protocol | Port range | Sources |
|---|---|---|---|
| Inbound | TCP | 22 | Your IP (or a bastion) — not `0.0.0.0/0` if avoidable |
| Inbound | TCP | 80 | All IPv4, All IPv6 |
| Inbound | TCP | 443 | All IPv4, All IPv6 |
| Outbound | All | All | All IPv4, All IPv6 (default) |

Do **not** add a rule for port `8000` — per `DEPLOYMENT.md`'s security section,
uvicorn binds to `127.0.0.1` only and is reached exclusively through Nginx on
the same machine. There's no reason for it to ever be internet-reachable, and
this firewall is a second layer of enforcement on top of that.

## 3. Point DNS at the Droplet

Create an **A record** for `py.ekict.com` → the Droplet's public IPv4 address.
If DigitalOcean is managing the domain's nameservers, this is under Networking
→ Domains; otherwise create the same A record with whatever registrar/DNS
provider hosts `ekict.com`. Give it a few minutes to propagate before running
`certbot` in the next stage (it does an HTTP challenge against the domain, so
DNS must resolve to this Droplet first).

## 4. Everything else: follow `DEPLOYMENT.md`

SSH into the Droplet and work through `DEPLOYMENT.md` steps 0–8, substituting:
- Domain: `py.ekict.com` (not the `py.orthosage.co` placeholder in that doc)
- Weights file: `hrnet_w32_ceph19_state_dict.pth` (not the generic `model.pth`
  placeholder) — transfer it exactly as covered earlier: `tar -czf
  weights.tar.gz weights/` locally, `scp`/`rsync` it over, `tar -xzf` on the
  Droplet, then point `MODEL_WEIGHTS_PATH` in the systemd unit at the
  extracted `.pth` file.
- Service name: `orthosage-ai` (used consistently throughout `DEPLOYMENT.md`'s
  systemd unit, `journalctl` commands, and troubleshooting table).

That covers: system packages, venv + CPU-only torch/torchvision install,
weights placement, the `orthosage-ai` systemd service, the Nginx reverse
proxy, and `certbot` for TLS.

## 5. DigitalOcean-specific extras worth doing

- **Swap file**: on the 2 GB Droplet especially, add swap so a memory spike
  during inference gets slowed down instead of triggering the OOM killer and
  crash-looping the service (the exact 502 failure mode covered in
  `DEPLOYMENT.md`'s troubleshooting table):
  ```bash
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  ```
- **Monitoring**: enable the free DigitalOcean Droplet monitoring agent
  (Droplet → Monitoring → Enable) to get CPU/memory graphs and optional
  alerts — useful for spotting whether a Droplet is undersized before it
  starts crash-looping under load.
- **Snapshots**: once the service is verified working end-to-end (`curl
  https://py.ekict.com/health` returns `{"status":"ok"}` and a real
  `/image-digitization` call returns correct landmarks), take a Droplet
  snapshot. Cheap insurance against having to redo this whole setup.

## 6. Final verification

Same as `DEPLOYMENT.md` step 6, against the real domain:
```bash
curl https://py.ekict.com/health
curl -X POST https://py.ekict.com/image-digitization -F "image=@/path/to/lateral-ceph.jpg"
```
The second call should return 19 landmarks with `confidence` values roughly in
the 8–20 range — if they look wildly different or land off-anatomy, the
weights file likely didn't load (check `sudo journalctl -u orthosage-ai -n 50`
for a state_dict or file-not-found error).
