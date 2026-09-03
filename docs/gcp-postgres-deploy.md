# Deploying PostgreSQL on GCP's Always Free Tier

This guide documents how the production database for pm-playbook was
moved from AWS RDS to a self-hosted PostgreSQL 16 instance on a Google
Cloud Compute Engine `e2-micro` VM, staying entirely inside GCP's
**Always Free** limits (a permanent free allowance, not a time-limited
trial credit).

It reflects what was actually run to provision the current production
database, not a hypothetical procedure — reuse it as-is to rebuild the
instance, or as a reference for the trade-offs involved.

## Why this approach, and what it costs you

A managed service like RDS gives you automatic backups, high
availability, failover, and a firewall-scoped network boundary out of the
box. A bare Compute Engine VM gives you none of that — you get PostgreSQL
running for free, and everything else is your responsibility:

- **No managed backups.** Nothing is scheduled by default. If you want
  point-in-time recovery, you need to set up your own `pg_dump` cron job
  (not included here — see "Backups" below).
- **No HA/failover.** One VM. If it goes down, the database is down.
- **Wider network exposure.** Streamlit Community Cloud has no fixed
  egress IPs to allowlist, so the firewall must accept connections from
  anywhere (`0.0.0.0/0`) on port 5432. Security rests entirely on
  mandatory TLS + `scram-sha-256` authentication, not network
  restriction.
- **1 GB/month free egress** (from North America) is the tightest Always
  Free limit that applies here — every query the app makes counts against
  it. Fine for a low-traffic app; a real constraint under load.

If none of that is acceptable, a managed service (Cloud SQL, RDS, etc.)
is the better choice — at a recurring cost.

## Prerequisites

- A Google Cloud account with a billing account linked (required by GCP
  even for Always Free resources — you are not charged as long as you
  stay within the limits below).
- The `gcloud` CLI, authenticated (`gcloud auth login`).
- No separate SSH key management needed — `gcloud compute ssh` handles
  key generation and propagation automatically.

## Cost checklist (stay inside Always Free)

- [ ] Machine type is exactly `e2-micro`
- [ ] Region is `us-west1`, `us-central1`, or `us-east1` (no other region
      qualifies — notably, no Canadian region is eligible)
- [ ] Boot disk ≤ 30 GB, standard persistent disk (not SSD)
- [ ] Only one Compute Engine VM instance running
- [ ] A billing budget alert is set as a tripwire (see step 1)

## 1. Create the project, link billing, enable the API

```bash
gcloud projects create pm-playbook-db --name="pm-playbook-db"
gcloud config set project pm-playbook-db
gcloud billing projects link pm-playbook-db --billing-account=<YOUR_BILLING_ACCOUNT_ID>
gcloud services enable compute.googleapis.com --project=pm-playbook-db
```

Set a budget alert scoped to just this project (adjust the currency code
to match your billing account's currency — check with
`gcloud billing accounts describe <ID>` first):

```bash
gcloud services enable billingbudgets.googleapis.com --project=pm-playbook-db
gcloud billing budgets create \
  --billing-account=<YOUR_BILLING_ACCOUNT_ID> \
  --display-name="pm-playbook-db \$1 tripwire" \
  --budget-amount=1.00USD \
  --filter-projects=projects/pm-playbook-db \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=1.0
```

**Verify:** `gcloud billing projects describe pm-playbook-db` shows
`billingEnabled: true`.

## 2. Create the VM

```bash
gcloud compute instances create pm-playbook-postgres \
  --project=pm-playbook-db \
  --zone=us-central1-a \
  --machine-type=e2-micro \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=10GB \
  --boot-disk-type=pd-standard \
  --tags=postgres-server
```

**Verify:** `gcloud compute instances list --project=pm-playbook-db`
shows the instance `RUNNING`, and note its external IP.

## 3. Open the Postgres port

Scoped to the VM's network tag, not the whole project or network:

```bash
gcloud compute firewall-rules create allow-postgres \
  --project=pm-playbook-db \
  --network=default \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:5432 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=postgres-server \
  --description="Allow inbound Postgres (TLS + scram-sha-256 required at the DB layer)"
```

**Verify:** `gcloud compute firewall-rules describe allow-postgres
--project=pm-playbook-db` shows the rule targeting `postgres-server`.

## 4. Install and harden PostgreSQL on the VM

Copy [`scripts/gcp_vm_postgres_setup.sh`](../scripts/gcp_vm_postgres_setup.sh)
to the VM and run it as root:

```bash
gcloud compute scp scripts/gcp_vm_postgres_setup.sh \
  pm-playbook-postgres:~/gcp_vm_postgres_setup.sh \
  --zone=us-central1-a --project=pm-playbook-db

gcloud compute ssh pm-playbook-postgres \
  --zone=us-central1-a --project=pm-playbook-db \
  --command="sudo bash ~/gcp_vm_postgres_setup.sh"
```

(If `gcloud compute scp` isn't reachable from where you're running this,
SSH in directly and paste the script's contents into a `cat > ... <<'EOF'`
heredoc instead — see the script itself for what it does.)

The script:

1. Installs PostgreSQL 16 from the official PGDG apt repo.
2. Generates a self-signed TLS certificate and enables `ssl = on`.
3. Rewrites `pg_hba.conf` to require `scram-sha-256` auth and `hostssl`
   (TLS) for every remote connection — there is no plaintext path.
4. Creates the `pm_playbook` database and a dedicated `pm_playbook_app`
   user with a randomly generated password.
5. Enables `ufw`, allowing only SSH and 5432, as defense-in-depth on top
   of the GCP firewall rule.

**The generated password is printed once, at the very end of the run, and
nowhere else.** Copy it immediately into a password manager or a local
untracked file — it cannot be recovered afterward (only reset, via
`ALTER ROLE pm_playbook_app WITH PASSWORD '...';`).

**Verify:**

```bash
gcloud compute ssh pm-playbook-postgres --zone=us-central1-a --project=pm-playbook-db \
  --command="sudo systemctl is-active postgresql && sudo ufw status"
```

## 5. Assemble the connection string

From the script's final output:

```dotenv
DATABASE_URL=postgresql://pm_playbook_app:<password>@<VM_EXTERNAL_IP>:5432/pm_playbook
POSTGRES_SSLMODE=require
```

Put this in a local, gitignored `.env` file — never commit it, and avoid
pasting the password into chat tools, tickets, or logs.

The VM's default IP is ephemeral and can change on restart. To pin it:

```bash
gcloud compute addresses create pm-playbook-postgres-ip \
  --project=pm-playbook-db --region=us-central1
gcloud compute instances delete-access-config pm-playbook-postgres \
  --project=pm-playbook-db --zone=us-central1-a --access-config-name="External NAT"
gcloud compute instances add-access-config pm-playbook-postgres \
  --project=pm-playbook-db --zone=us-central1-a \
  --access-config-name="External NAT" \
  --address="$(gcloud compute addresses describe pm-playbook-postgres-ip --project=pm-playbook-db --region=us-central1 --format='value(address)')"
```

A reserved static IP is free as long as it stays attached to a running
instance.

**Verify:** a raw TCP check confirms the port is reachable without needing
credentials:

```bash
timeout 5 bash -c 'cat < /dev/null > /dev/tcp/<VM_EXTERNAL_IP>/5432' \
  && echo reachable || echo "not reachable"
```

With `psql` installed locally:

```bash
psql "$DATABASE_URL?sslmode=require" -c '\conninfo'
```

## 6. Initialize the schema

`pm_playbook/db_prep.py` only depends on `sqlalchemy` and
`psycopg2-binary` — it does not need the full project dependency set
(PyTorch, Transformers, etc. pulled in by a plain `uv sync`). A scoped
venv is much faster if you don't already have the full project installed:

```bash
uv venv /tmp/db-prep-venv
uv pip install --python /tmp/db-prep-venv/bin/python "psycopg2-binary>=2.9.12" "sqlalchemy>=2.0.51"
set -a; source .env; set +a
/tmp/db-prep-venv/bin/python -m pm_playbook.db_prep
```

Or simply `uv run python -m pm_playbook.db_prep` if you already have the
full environment synced for other work.

**Verify:** tables exist and the connection is actually encrypted (not
just that the server supports TLS):

```python
from pm_playbook.db import get_engine
from sqlalchemy import text

with get_engine().connect() as conn:
    tables = conn.execute(text(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
    )).fetchall()
    print(tables)  # expect conversations, feedback

    ssl_row = conn.execute(text(
        "SELECT ssl, version FROM pg_stat_ssl JOIN pg_stat_activity USING (pid) "
        "WHERE pid = pg_backend_pid()"
    )).fetchone()
    print(ssl_row)  # expect (True, 'TLSv1.3')
```

## 7. Migrating existing data (skip if starting fresh)

If there's production data in the old database worth keeping:

```bash
pg_dump -Fc "$OLD_DATABASE_URL" > rds_export.dump
pg_restore -d "$DATABASE_URL" rds_export.dump
```

**Verify:** row counts match between old and new
(`SELECT count(*) FROM conversations;` on both).

## 8. Backups

**Not configured by default in this setup.** Unlike RDS, this VM has no
automatic snapshots. If you want a nightly `pg_dump` cron job with a
retention window, that needs to be added deliberately — decide the
retention period and where dumps are stored (the VM's own disk is not a
real backup, since it fails together with the database) before wiring
anything up.

## 9. Point the application at the new database

In Streamlit Community Cloud: **Settings → Secrets**, set `DATABASE_URL`
(and `POSTGRES_SSLMODE=require`) to the values from step 5, save, and let
it redeploy.

**Verify:** the app loads, and a test question through the UI produces a
new row in `conversations` — check via the Monitoring page or by querying
directly, comparing the row count before and after.

## Rollback

Until the old database is decommissioned, reverting the
`DATABASE_URL` secret back to the old connection string fully rolls back
to it — nothing about this setup is destructive to the old instance.

## Decommissioning the old database

Only after the new instance has been live and stable for a few days.
This is a deliberate, separately-confirmed action — not something to
bundle into the initial cutover. For AWS RDS:

```bash
aws rds delete-db-instance --db-instance-identifier <your-instance-id> \
  --final-db-snapshot-identifier pm-playbook-final-snapshot
```

Keep the final snapshot for a while as a safety net before deleting it
too.
