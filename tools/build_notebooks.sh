#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NOTEBOOKS_DIR="${PROJECT_ROOT}/notebooks"
FORCE_ALL=false

usage() {
  cat <<'EOF'
Uso: ./build_notebooks.sh [--all] [--notebooks-dir <path>]

Opzioni:
  --all                    Converte tutti i file .py trovati.
  --notebooks-dir <path>   Directory da cui cercare i file .py (default: ../notebooks).
  -h, --help               Mostra questo messaggio.

Comportamento di default:
  Converte solo i .py che sembrano notebook Jupytext in formato percent,
  rilevati tramite metadati jupytext o celle con marker '# %%'.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      FORCE_ALL=true
      shift
      ;;
    --notebooks-dir)
      if [[ $# -lt 2 ]]; then
        echo "Errore: --notebooks-dir richiede un percorso." >&2
        usage
        exit 1
      fi
      NOTEBOOKS_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Errore: opzione non riconosciuta: $1" >&2
      usage
      exit 1
      ;;
  esac
done

declare -a JUPYTEXT_RUNNER=()

set_jupytext_runner() {
  local local_venv_win="${PROJECT_ROOT}/.venv/Scripts/python.exe"
  local local_venv_unix="${PROJECT_ROOT}/.venv/bin/python"

  if [[ -f "${local_venv_win}" ]] && "${local_venv_win}" -c "import jupytext" >/dev/null 2>&1; then
    JUPYTEXT_RUNNER=("${local_venv_win}" -m jupytext)
    return
  fi

  if [[ -x "${local_venv_unix}" ]] && "${local_venv_unix}" -c "import jupytext" >/dev/null 2>&1; then
    JUPYTEXT_RUNNER=("${local_venv_unix}" -m jupytext)
    return
  fi

  if command -v python >/dev/null 2>&1 && python -c "import jupytext" >/dev/null 2>&1; then
    JUPYTEXT_RUNNER=(python -m jupytext)
    return
  fi

  if command -v python3 >/dev/null 2>&1 && python3 -c "import jupytext" >/dev/null 2>&1; then
    JUPYTEXT_RUNNER=(python3 -m jupytext)
    return
  fi

  if command -v py >/dev/null 2>&1 && py -c "import jupytext" >/dev/null 2>&1; then
    JUPYTEXT_RUNNER=(py -m jupytext)
    return
  fi

  if command -v jupytext.exe >/dev/null 2>&1; then
    JUPYTEXT_RUNNER=(jupytext.exe)
    return
  fi

  if command -v jupytext >/dev/null 2>&1; then
    JUPYTEXT_RUNNER=(jupytext)
    return
  fi

  echo "Errore: impossibile trovare un comando funzionante per jupytext." >&2
  echo "Installa jupytext nell'ambiente Python attivo (es. python -m pip install jupytext)." >&2
  exit 1
}

is_percent_notebook_source() {
  local py_file="$1"

  if grep -qE "format_name:[[:space:]]*percent" "${py_file}"; then
    return 0
  fi

  if grep -qE '^[[:space:]]*#[[:space:]]*%%([[:space:]]*\[markdown\])?[[:space:]]*$' "${py_file}"; then
    return 0
  fi

  return 1
}

run_jupytext() {
  "${JUPYTEXT_RUNNER[@]}" "$@"
}

if [[ ! -d "${NOTEBOOKS_DIR}" ]]; then
  echo "Errore: directory notebooks non trovata in ${NOTEBOOKS_DIR}" >&2
  exit 1
fi

set_jupytext_runner
echo "Directory notebook: ${NOTEBOOKS_DIR}"
echo "Runner jupytext: ${JUPYTEXT_RUNNER[*]}"

converted=0
skipped=0
failed=0
found=0

while IFS= read -r -d '' py_file; do
  found=$((found + 1))
  if [[ "${FORCE_ALL}" == true ]] || is_percent_notebook_source "${py_file}"; then
    ipynb_file="${py_file%.py}.ipynb"
    echo "Converto: ${py_file} -> ${ipynb_file}"
    if run_jupytext --to ipynb "${py_file}" -o "${ipynb_file}"; then
      converted=$((converted + 1))
    else
      echo "Errore: conversione fallita per ${py_file}" >&2
      failed=$((failed + 1))
    fi
  else
    skipped=$((skipped + 1))
  fi
done < <(find "${NOTEBOOKS_DIR}" -type f -name "*.py" -print0)

echo "Conversione completata."
echo "File .py trovati: ${found}"
echo "Notebook convertiti: ${converted}"
echo "File .py saltati: ${skipped}"
echo "Conversioni fallite: ${failed}"

if [[ ${failed} -gt 0 ]]; then
  exit 1
fi
