#!/usr/bin/env bash
#
# gcp_vm_postgres_setup.sh
#
# Provisions PostgreSQL 16 on a fresh Debian 12 Compute Engine VM as a
# self-hosted replacement for AWS RDS. Run as root (e.g. via
# `gcloud compute ssh <vm> --command="sudo bash gcp_vm_postgres_setup.sh"`)
# on a VM that has already been created and tagged for the
# `allow-postgres` firewall rule (TCP 5432 inbound).
#
# What it does:
#   1. Installs PostgreSQL 16 from the official PGDG apt repo.
#   2. Generates a self-signed TLS cert and enables `ssl = on`.
#   3. Rewrites pg_hba.conf to require scram-sha-256 auth and TLS
#      (hostssl) for all remote connections.
#   4. Creates the `pm_playbook` database and a dedicated
#      `pm_playbook_app` user with a randomly generated password.
#   5. Enables ufw as defense-in-depth on top of the GCP firewall rule.
#
# The generated password is printed ONCE at the end of the run. Copy it
# immediately — it is not stored anywhere on disk by this script and is
# not recoverable afterward (you'd need to ALTER USER ... PASSWORD to
# reset it).

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "This script must be run as root (e.g. with sudo)." >&2
  exit 1
fi

PG_VERSION="16"
APP_DB="pm_playbook"
APP_USER="pm_playbook_app"
PG_CONF_DIR="/etc/postgresql/${PG_VERSION}/main"
SSL_DIR="${PG_CONF_DIR}/ssl"

echo "==> Detecting external IP (used only for the TLS cert subject and final output)"
EXTERNAL_IP="$(curl -s -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip')"
echo "    External IP: ${EXTERNAL_IP}"

echo "==> Installing prerequisites"
apt-get update -qq
apt-get install -y -qq curl ca-certificates gnupg lsb-release openssl ufw >/dev/null

echo "==> Adding the PGDG apt repo for PostgreSQL ${PG_VERSION}"
install -d /usr/share/postgresql-common/pgdg
curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
  https://www.postgresql.org/media/keys/ACCC4CF8.asc
echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
  > /etc/apt/sources.list.d/pgdg.list
apt-get update -qq

echo "==> Installing PostgreSQL ${PG_VERSION}"
apt-get install -y -qq "postgresql-${PG_VERSION}" >/dev/null

echo "==> Generating a self-signed TLS certificate"
mkdir -p "${SSL_DIR}"
openssl req -new -x509 -days 3650 -nodes \
  -subj "/CN=${EXTERNAL_IP}" \
  -out "${SSL_DIR}/server.crt" \
  -keyout "${SSL_DIR}/server.key"
chown postgres:postgres "${SSL_DIR}/server.key" "${SSL_DIR}/server.crt"
chmod 600 "${SSL_DIR}/server.key"
chmod 644 "${SSL_DIR}/server.crt"

echo "==> Configuring postgresql.conf (TLS + listen on all interfaces)"
cat >> "${PG_CONF_DIR}/postgresql.conf" <<EOF

# --- pm_playbook: added by gcp_vm_postgres_setup.sh ---
listen_addresses = '*'
ssl = on
ssl_cert_file = '${SSL_DIR}/server.crt'
ssl_key_file = '${SSL_DIR}/server.key'
password_encryption = scram-sha-256
EOF

echo "==> Rewriting pg_hba.conf (scram-sha-256 + mandatory TLS for remote connections)"
cat > "${PG_CONF_DIR}/pg_hba.conf" <<EOF
# pm_playbook: replaced by gcp_vm_postgres_setup.sh
# TYPE  DATABASE  USER  ADDRESS       METHOD

# Local socket connections (used by the postgres superuser locally)
local   all       all                 peer

# Loopback connections
hostssl all       all   127.0.0.1/32  scram-sha-256
hostssl all       all   ::1/128       scram-sha-256

# Remote connections: TLS required, scram-sha-256 auth required
hostssl all       all   0.0.0.0/0     scram-sha-256
hostssl all       all   ::/0          scram-sha-256

# Everything else (including any plaintext "host" attempt) is rejected
# implicitly because no matching line exists.
EOF

echo "==> Creating database and app user"
APP_PASSWORD="$(openssl rand -base64 24)"
sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${APP_USER}') THEN
    CREATE ROLE ${APP_USER} WITH LOGIN PASSWORD '${APP_PASSWORD}';
  ELSE
    ALTER ROLE ${APP_USER} WITH PASSWORD '${APP_PASSWORD}';
  END IF;
END
\$\$;
SQL
sudo -u postgres psql -v ON_ERROR_STOP=1 -tc \
  "SELECT 1 FROM pg_database WHERE datname = '${APP_DB}'" | grep -q 1 \
  || sudo -u postgres createdb -O "${APP_USER}" "${APP_DB}"

echo "==> Restarting PostgreSQL"
systemctl enable postgresql >/dev/null
systemctl restart postgresql

echo "==> Enabling ufw as defense-in-depth on top of the GCP firewall rule"
ufw allow OpenSSH >/dev/null
ufw allow 5432/tcp >/dev/null
ufw --force enable

echo ""
echo "=================================================================="
echo " pm_playbook PostgreSQL setup complete"
echo "=================================================================="
echo ""
echo " Save this now — the password is not stored anywhere and will not"
echo " be shown again:"
echo ""
echo "   DATABASE_URL=postgresql://${APP_USER}:${APP_PASSWORD}@${EXTERNAL_IP}:5432/${APP_DB}"
echo "   POSTGRES_SSLMODE=require"
echo ""
echo "=================================================================="
