#!/bin/bash
# Snapshot what the frontend's pure utils produce for heart + NHANES, for a before/after diff.
# usage: snap.sh <workdir> [frontend_src_dir]
#   <workdir>/schemas.json must exist (dump_schemas.py); writes <workdir>/out.json
# The utils import without extensions, so they are copied as .mjs with fixed imports.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(cd "$1" && pwd)
SRC=${2:-$HERE/../../../../frontend/src}
M=$WORK/mods; rm -rf "$M"; mkdir -p "$M"
for f in "$SRC"/utils/*.js "$SRC"/constants/*.js; do
  b=$(basename "$f" .js)
  sed -E "s#from '(\.\.?/[^']+)'#from '\1.mjs'#; s#from '\.\./constants/([^.']+)\.mjs'#from './\1.mjs'#; s#from '\.\./utils/([^.']+)\.mjs'#from './\1.mjs'#" "$f" > "$M/$b.mjs"
done
cp "$HERE/harness.mjs" "$HERE/screens.mjs" "$M/"
(cd "$M" && node harness.mjs "$WORK/schemas.json" > "$WORK/out.json")
echo "wrote $WORK/out.json"
