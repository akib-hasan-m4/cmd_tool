#!/usr/bin/env bash
# Regenerate docs/architecture.pdf from docs/architecture.html.
#
# architecture.html is authored as Artifact-shaped content (no <!doctype>/<html>/
# <head>/<body> wrapper -- the Artifact host supplies those). Chromium needs a
# full document, so we wrap it here into docs/.build/ and print that.
#
# The intermediate file must live under $HOME: chromium here is the snap build,
# and snap confinement blocks reads from /tmp.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
src="$here/architecture.html"
build="$here/.build"
tmp="$build/architecture.full.html"
out="$here/architecture.pdf"

mkdir -p "$build"

{
  printf '<!doctype html>\n<html lang="en">\n<head>\n'
  printf '<meta charset="utf-8">\n'
  printf '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
  printf '<style>body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%%}[hidden]{display:none!important}</style>\n'
  printf '</head>\n<body>\n'
  cat "$src"
  printf '\n</body>\n</html>\n'
} > "$tmp"

chromium --headless --disable-gpu --no-sandbox \
         --no-pdf-header-footer \
         --virtual-time-budget=20000 \
         --print-to-pdf="$out" "file://$tmp"

echo "wrote $out ($(du -h "$out" | cut -f1))"
