#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Optional per-user TeX support cache (beamer and metropolis live there on the authors' machine).
anlp_texmf_cache="${ANLP_TEXMF_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/anlp-report-tools/texmf}"
if [[ -d "$anlp_texmf_cache" ]]; then
  export TEXMFHOME="$anlp_texmf_cache"
fi
mkdir -p build
for pass in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex >build/pass$pass.log 2>&1 || { tail -40 build/pass$pass.log; exit 1; }
done
cp build/main.pdf main.pdf
printf '%s\n' 'Built main.pdf'
