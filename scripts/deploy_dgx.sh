#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.dgx"
RUNTIME_DIR="${ROOT_DIR}/.runtime"
POSTGRES_ENV_FILE="${RUNTIME_DIR}/postgres.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  printf 'Missing %s\n' "${ENV_FILE}" >&2
  exit 1
fi

mkdir -p "${RUNTIME_DIR}"

set -a
source "${ENV_FILE}"
set +a

PDFS_HOST_DIR="${PDFS_HOST_DIR:-${ROOT_DIR}/pdfs}"
CACHE_HOST_DIR="${CACHE_HOST_DIR:-${ROOT_DIR}/text_cache}"
LOGS_HOST_DIR="${LOGS_HOST_DIR:-${ROOT_DIR}/logs}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${ROOT_DIR}/model_cache}"

mkdir -p "${PDFS_HOST_DIR}" "${CACHE_HOST_DIR}" "${LOGS_HOST_DIR}" "${MODEL_CACHE_DIR}"

required_vars=(
  POSTGRES_USER
  POSTGRES_PASSWORD
  POSTGRES_DB
  DB_PASSWORD
  JWT_SECRET_KEY
  LLM_MODEL
  LLM_BASE_URL
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    printf 'Required variable %s is empty in %s\n' "${var_name}" "${ENV_FILE}" >&2
    exit 1
  fi

  if [[ "${!var_name}" == REPLACE_WITH_* ]]; then
    printf 'Replace placeholder %s in %s before deploying\n' "${var_name}" "${ENV_FILE}" >&2
    exit 1
  fi
done

if [[ "${POSTGRES_PASSWORD}" != "${DB_PASSWORD}" ]]; then
  printf 'POSTGRES_PASSWORD and DB_PASSWORD must be identical in %s\n' "${ENV_FILE}" >&2
  exit 1
fi

cat > "${POSTGRES_ENV_FILE}" <<EOF
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=${POSTGRES_DB}
EOF

cd "${ROOT_DIR}"

docker compose --env-file .env.dgx up -d --build

printf 'Waiting for API health...\n'
for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

if ! curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  printf 'API did not become healthy in time. Check docker compose logs.\n' >&2
  exit 1
fi

docker compose --env-file .env.dgx exec -T api python -m api.seed_admin

printf 'Checking vLLM readiness (non-fatal)...\n'
if curl -fsS http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
  printf 'vLLM is ready.\n'
else
  printf 'vLLM is still warming up. The API/frontend are up, but RAG answers will fail until the model endpoint responds.\n'
  printf 'Check progress with:\n'
  printf '  docker compose --env-file .env.dgx ps\n'
  printf '  docker compose --env-file .env.dgx logs --tail=200 vllm-qwen\n'
  printf '  curl -fsS http://127.0.0.1:8001/v1/models\n'
fi

printf '\nDGX deployment is up.\n'
printf 'Frontend: http://127.0.0.1:3000\n'
printf 'API docs: http://127.0.0.1:8000/docs\n'
printf '\nIf you already have PDFs in ./pdfs, import them into the app database with a department-specific command, for example:\n'
printf 'docker compose --env-file .env.dgx exec -T api python import_legacy_pdfs.py --uploaded-by-email admin@tunisie-electronique.com --department-id commerciale --reindex\n'
