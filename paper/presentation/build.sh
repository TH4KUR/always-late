#!/usr/bin/env bash
# Builds the presentation PDF. `bash build.sh final` refuses to build while a result slide is still marked pending.
set -euo pipefail
cd -- "$(dirname -- "$0")"; # Optional per-user TeX support cache (the font encodings of psnfss), as in the report's build.sh
anlp_texmf_cache="${ANLP_TEXMF_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/anlp-report-tools/texmf}"
if [[ -d "$anlp_texmf_cache" ]]; then export TEXMFHOME="$anlp_texmf_cache"; fi
mkdir -p build
if [ "${1:-}" = "final" ] && grep -q '^\\pending{' slides.tex; then echo "result slides are still pending:"; grep -n '^\\pending{' slides.tex | cut -c1-90; exit 1; fi
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build slides.tex >build/pass1.log 2>&1 || { tail -40 build/pass1.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build slides.tex >build/pass2.log 2>&1 || { tail -40 build/pass2.log; exit 1; }
cp build/slides.pdf AlwaysLate-Final-Presentation.pdf; echo "Built AlwaysLate-Final-Presentation.pdf ($(pdfinfo AlwaysLate-Final-Presentation.pdf 2>/dev/null | grep Pages || echo pages unknown))"
