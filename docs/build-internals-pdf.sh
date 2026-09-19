#!/usr/bin/env bash
# Regenerate docs/internals.pdf from docs/internals.html.
#
# internals.html is authored as Artifact-shaped content (no <!doctype>/<html>/
# <head>/<body> wrapper -- the Artifact host supplies those). Chromium needs a
# full document, so we wrap it here into docs/.build/ and print that.
#
# The intermediate file must live under $HOME: chromium here is the snap build,
# and snap confinement blocks reads from /tmp.
#
# Page size comes from the @page rule in internals.html (105mm x 187mm, a 9:16
# phone page), so no --print-to-pdf paper flags are passed here. Verify after a
# change with: pdfinfo docs/internals.pdf | grep 'Page size'.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
src="$here/internals.html"
build="$here/.build"
tmp="$build/internals.full.html"
out="$here/internals.pdf"

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
