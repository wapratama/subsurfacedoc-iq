"""
monitoring/grafana/init.py

Auto-provisions Grafana datasource + dashboard via HTTP API.
Run once after docker compose up -d.

Mirrors the fitness-assistant pattern from Module 7.
Without this, reviewers must manually configure Grafana — 
which breaks the "one-command reproducibility" criterion.
"""
import os
import json
import time
import requests
from pathlib import Path

GRAFANA_URL  = os.environ.get("GRAFANA_URL",  "http://localhost:3000")
GRAFANA_USER = os.environ.get("GRAFANA_USER", "admin")
GRAFANA_PASS = os.environ.get("GRAFANA_PASSWORD", "admin")
AUTH         = (GRAFANA_USER, GRAFANA_PASS)

PG_HOST  = os.environ.get("POSTGRES_HOST",     "postgres")
PG_DB    = os.environ.get("POSTGRES_DB",        "subsurface_monitoring")
PG_USER  = os.environ.get("POSTGRES_USER",      "admin")
PG_PASS  = os.environ.get("POSTGRES_PASSWORD",  "admin")


def wait_for_grafana(retries: int = 15):
    """Wait until Grafana is healthy before provisioning."""
    for i in range(retries):
        try:
            r = requests.get(f"{GRAFANA_URL}/api/health", timeout=3)
            if r.status_code == 200:
                print("✅ Grafana is healthy")
                return
        except Exception:
            pass
        print(f"  Waiting for Grafana... ({i+1}/{retries})")
        time.sleep(3)
    raise RuntimeError("Grafana did not become healthy")


def provision_datasource():
    """Create Postgres datasource in Grafana."""
    payload = {
        "name":   "SubsurfaceIQ-Postgres",
        "type":   "postgres",
        "url":    f"{PG_HOST}:5432",
        "user":   PG_USER,
        "secureJsonData": {"password": PG_PASS},
        "jsonData": {
            "database":        PG_DB,
            "sslmode":         "disable",
            "postgresVersion": 1500,
            "timescaledb":     False,
        },
        "access":    "proxy",
        "isDefault": True,
    }
    r = requests.post(
        f"{GRAFANA_URL}/api/datasources",
        auth=AUTH,
        json=payload,
    )
    if r.status_code in (200, 409):  # 409 = already exists
        print(f"✅ Datasource provisioned (status {r.status_code})")
    else:
        print(f"⚠️  Datasource: {r.status_code} {r.text[:200]}")


def provision_dashboard():
    """Import the SubsurfaceIQ dashboard JSON."""
    dashboard_path = (
        Path(__file__).parent / "dashboards" / "subsurfaceiq.json"
    )
    if not dashboard_path.exists():
        print(f"⚠️  Dashboard JSON not found: {dashboard_path}")
        return

    dashboard = json.loads(dashboard_path.read_text())
    payload = {
        "dashboard": dashboard,
        "overwrite": True,
        "folderId":  0,
    }
    r = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db",
        auth=AUTH,
        json=payload,
    )
    if r.status_code == 200:
        result = r.json()
        print(f"✅ Dashboard provisioned: {result.get('url', '')}")
    else:
        print(f"⚠️  Dashboard: {r.status_code} {r.text[:200]}")


def run():
    print("=== Grafana Provisioning ===")
    wait_for_grafana()
    provision_datasource()
    provision_dashboard()
    print("\n✅ Grafana ready at http://localhost:3000")
    print(f"   Login: {GRAFANA_USER} / {GRAFANA_PASS}")


if __name__ == "__main__":
    run()