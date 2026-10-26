#!/usr/bin/env python3
"""Create (or reuse) a remotely-managed Cloudflare Tunnel for the panel and write its token to deploy/.env.

Idempotent: re-running reuses the tunnel (by name) and the DNS record. Standard library only.

    python3 deploy/cloudflare_setup.py            # dry run: shows what would change
    python3 deploy/cloudflare_setup.py --apply    # create / update tunnel, ingress, DNS; write CF_TUNNEL_TOKEN
    python3 deploy/cloudflare_setup.py --apply --yes   # also re-point a hostname that already points elsewhere

Reads deploy/.env: CF_API_TOKEN (Account > Cloudflare Tunnel > Edit, Zone > DNS > Edit), CF_ACCOUNT_ID (optional
if the token sees exactly one account), CF_HOSTNAME (e.g. my-code.example.com), CF_TUNNEL_NAME, and optionally
CF_SERVICE (default http://lab:8081 = the lab container on the compose network).
Then: docker compose -f deploy/docker-compose.yml --profile tunnel up -d cloudflared
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.cloudflare.com/client/v4"
ENV = Path(__file__).resolve().parent / ".env"


def read_env(path: Path) -> dict:
    out = {}
    if path.exists():
        for ln in path.read_text().splitlines():
            if "=" in ln and not ln.lstrip().startswith("#"):
                k, v = ln.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def write_env_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    for m, ln in enumerate(lines):
        if ln.split("=", 1)[0].strip() == key:
            lines[m] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")
    os.chmod(path, 0o600)


class CF:
    def __init__(self, token: str):
        self.token = token

    def call(self, method: str, path: str, body: dict | None = None, query: str = ""):
        req = urllib.request.Request(f"{API}{path}{'?' + query if query else ''}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                res = json.loads(r.read())
        except urllib.error.HTTPError as e:
            res = json.loads(e.read() or b"{}")
        if not res.get("success"):
            sys.exit(f"Cloudflare API {method} {path} failed: {res.get('errors')}")
        return res["result"]


def zone_of(cf: CF, hostname: str) -> dict:
    """The zone the hostname belongs to (longest matching suffix)."""
    parts = hostname.split(".")
    for m in range(len(parts) - 1):
        zs = cf.call("GET", "/zones", query=f"name={'.'.join(parts[m:])}")
        if zs:
            return zs[0]
    sys.exit(f"no Cloudflare zone found for {hostname} (is the domain on this account, and may the token read it?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="make the changes (default: dry run)")
    ap.add_argument("--yes", action="store_true", help="allow re-pointing a hostname that currently points elsewhere")
    a = ap.parse_args()
    env = {**read_env(ENV), **{k: v for k, v in os.environ.items() if k.startswith("CF_")}}
    for k in ("CF_API_TOKEN", "CF_HOSTNAME"):
        if not env.get(k):
            sys.exit(f"{k} is empty: set it in {ENV}")
    host, name = env["CF_HOSTNAME"], env.get("CF_TUNNEL_NAME") or "mycode"
    service = env.get("CF_SERVICE") or "http://lab:8081"
    cf = CF(env["CF_API_TOKEN"])
    dry = "" if a.apply else "[dry run] "

    acc = env.get("CF_ACCOUNT_ID")
    if not acc:
        accs = cf.call("GET", "/accounts")
        if len(accs) != 1:
            sys.exit(f"the token sees {len(accs)} accounts: set CF_ACCOUNT_ID")
        acc = accs[0]["id"]

    # 1. tunnel (reuse by name)
    found = cf.call("GET", f"/accounts/{acc}/cfd_tunnel", query=f"name={name}&is_deleted=false")
    if found:
        tid = found[0]["id"]
        print(f"tunnel '{name}' exists: {tid}")
    elif a.apply:
        t = cf.call("POST", f"/accounts/{acc}/cfd_tunnel", {"name": name, "config_src": "cloudflare",
                                                             "tunnel_secret": base64.b64encode(secrets.token_bytes(32)).decode()})
        tid = t["id"]
        print(f"tunnel '{name}' created: {tid}")
    else:
        tid = "<new>"
        print(f"{dry}would create tunnel '{name}'")

    # 2. ingress: hostname -> the panel; everything else 404
    ingress = {"config": {"ingress": [{"hostname": host, "service": service}, {"service": "http_status:404"}]}}
    if a.apply:
        cf.call("PUT", f"/accounts/{acc}/cfd_tunnel/{tid}/configurations", ingress)
    print(f"{dry}ingress: {host} -> {service}")

    # 3. DNS: proxied CNAME <host> -> <tunnel id>.cfargotunnel.com
    zone = zone_of(cf, host)
    target = f"{tid}.cfargotunnel.com"
    recs = cf.call("GET", f"/zones/{zone['id']}/dns_records", query=f"name={host}")
    rec = {"type": "CNAME", "name": host, "content": target, "proxied": True}
    if not recs:
        if a.apply:
            cf.call("POST", f"/zones/{zone['id']}/dns_records", rec)
        print(f"{dry}DNS: create CNAME {host} -> {target}")
    elif recs[0].get("content") == target:
        print(f"DNS: {host} already points to this tunnel")
    else:
        print(f"DNS: {host} currently points to {recs[0].get('type')} {recs[0].get('content')}")
        if not a.yes:
            sys.exit("refusing to re-point it (live traffic would move to this machine); re-run with --yes to do so")
        if a.apply:
            cf.call("PUT", f"/zones/{zone['id']}/dns_records/{recs[0]['id']}", rec)
        print(f"{dry}DNS: re-point {host} -> {target}")

    # 4. connector token -> deploy/.env (the cloudflared container reads it)
    if a.apply:
        tok = cf.call("GET", f"/accounts/{acc}/cfd_tunnel/{tid}/token")
        write_env_value(ENV, "CF_TUNNEL_TOKEN", tok)
        print(f"CF_TUNNEL_TOKEN written to {ENV} (mode 600)")
        print("next: docker compose -f deploy/docker-compose.yml --profile tunnel up -d cloudflared")
    else:
        print(f"{dry}would write CF_TUNNEL_TOKEN to {ENV}; run again with --apply")


if __name__ == "__main__":
    main()
