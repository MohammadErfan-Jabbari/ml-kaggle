#!/usr/bin/env bash
set -euo pipefail

name="${1:-main}"

cd "$(dirname "$0")"

tex="${name}.tex"
pdf="${name}.pdf"
png="${name}.png"

if [[ ! -f "$tex" ]]; then
  echo "Missing LaTeX source: $tex" >&2
  exit 1
fi

latexmk -pdf -interaction=nonstopmode -halt-on-error "$tex"

if ! command -v pdftoppm >/dev/null 2>&1; then
  echo "Missing 'pdftoppm' (poppler-utils). Cannot render PNG preview." >&2
  exit 1
fi

pdftoppm -png -singlefile -r 150 "$pdf" "$name" >/dev/null
echo "Wrote: $pdf"
echo "Wrote: $png"

