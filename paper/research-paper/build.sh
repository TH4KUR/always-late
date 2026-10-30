#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Optional per-user TeX support cache; standard complete TeX installs need none.
anlp_texmf_cache="${ANLP_TEXMF_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/anlp-report-tools/texmf}"
if [[ -d "$anlp_texmf_cache" ]]; then
  export TEXMFHOME="$anlp_texmf_cache"
fi
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex >build/pass1.log 2>&1 || { tail -50 build/pass1.log; exit 1; }
bibtex build/main >build/bibtex.log 2>&1 || { cat build/bibtex.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex >build/pass2.log 2>&1 || { tail -50 build/pass2.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex >build/pass3.log 2>&1 || { tail -50 build/pass3.log; exit 1; }
cp build/main.pdf Feature-Survival-and-Intervention-Preservation.pdf
printf '%s\n' 'Built Feature-Survival-and-Intervention-Preservation.pdf'
