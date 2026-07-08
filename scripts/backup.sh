#!/bin/bash
# backup.sh — Automated daily backup for Tunisie Electronique RAG
set -euo pipefail

BACKUP_DIR=/opt/rag/backups/$(date +%Y%m%d)
mkdir -p "$BACKUP_DIR"

echo "[$(date)] Starting backup..."

# PostgreSQL dump
docker compose -f /opt/rag/docker-compose.prod.yml exec -T postgres \
    pg_dump -U rag_user rag_enterprise > "$BACKUP_DIR/pg_dump.sql"
echo "  ✓ PostgreSQL dumped"

# Qdrant snapshot
curl -s -X POST http://localhost:6333/collections/rag_docs/snapshots > /dev/null
echo "  ✓ Qdrant snapshot created"

# Uploaded PDFs
cp -r /opt/rag/app_data/pdfs "$BACKUP_DIR/pdfs/" 2>/dev/null || true
echo "  ✓ PDFs copied"

# Retain 7 days
find /opt/rag/backups -maxdepth 1 -mtime +7 -exec rm -rf {} \;
echo "  ✓ Old backups cleaned"

echo "[$(date)] Backup complete → $BACKUP_DIR"
