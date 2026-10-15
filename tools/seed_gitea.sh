#!/usr/bin/env bash
# Start a local Gitea (SQLite) and seed the repository used by the real-app
# fixtures and the Phase 5 check. Passwords are throwaway and local only.
# Usage: tools/seed_gitea.sh [data-dir]   (default: artifacts/realapp)
set -euo pipefail
DATA=${1:-artifacts/realapp}
IMAGE=docker.io/gitea/gitea:1.22.6-rootless
PORT=3300
mkdir -p "$DATA/gitea-data" "$DATA/gitea-config"
podman rm -f vga-gitea >/dev/null 2>&1 || true
# keep-id maps the container's uid 1000 to the host user so the volumes are writable.
podman run -d --name vga-gitea --userns=keep-id -p 127.0.0.1:$PORT:3000 \
  -e GITEA__database__DB_TYPE=sqlite3 -e GITEA__security__INSTALL_LOCK=true \
  -e GITEA__server__ROOT_URL=http://127.0.0.1:$PORT/ -e GITEA__service__DISABLE_REGISTRATION=true \
  -e GITEA__server__OFFLINE_MODE=true \
  -v "$PWD/$DATA/gitea-data:/var/lib/gitea:Z" -v "$PWD/$DATA/gitea-config:/etc/gitea:Z" "$IMAGE" >/dev/null
timeout 90 bash -c "until curl -sf http://127.0.0.1:$PORT/api/healthz >/dev/null; do :; done"

user() { podman exec vga-gitea gitea admin user create --username "$1" --password "$2" --email "$1@example.test" --must-change-password=false ${3:-}; }
user owner ownerpass123 --admin
user viewer viewerpass123
user maintainer maintainerpass123

A=(-sf -u owner:ownerpass123 -H Content-Type:application/json)
API=http://127.0.0.1:$PORT/api/v1
R=$API/repos/owner/launch-plan
b64() { printf '%s' "$1" | base64 -w0; }
sha() { curl "${A[@]}" "$R/contents/README.md?ref=$1" | python3 -c 'import sys,json;print(json.load(sys.stdin)["sha"])'; }

curl "${A[@]}" -X POST $API/user/repos -d '{"name":"launch-plan","description":"Q3 launch planning","auto_init":true}' -o /dev/null
curl "${A[@]}" -X PUT $R/collaborators/viewer -d '{"permission":"read"}' -o /dev/null
curl "${A[@]}" -X PUT $R/collaborators/maintainer -d '{"permission":"write"}' -o /dev/null
for label in bug:ee0701 enhancement:84b6eb documentation:0075ca; do
  curl "${A[@]}" -X POST $R/labels -d "{\"name\":\"${label%%:*}\",\"color\":\"#${label##*:}\"}" -o /dev/null
done
curl "${A[@]}" -X POST $R/milestones -d '{"title":"Q3 release"}' -o /dev/null
for title in "Draft launch brief" "Export brief as PDF" "Collect reviewer feedback" "Update regional forecast"; do
  curl "${A[@]}" -X POST $R/issues -d "{\"title\":\"$title\",\"body\":\"Tracking: $title\"}" -o /dev/null
done
# One clean pull request and one that conflicts with main.
curl "${A[@]}" -X POST $R/branches -d '{"new_branch_name":"feature-brief","old_branch_name":"main"}' -o /dev/null
curl "${A[@]}" -X POST $R/contents/brief.md -d "{\"content\":\"$(b64 '# Launch brief')\",\"message\":\"Add brief\",\"branch\":\"feature-brief\"}" -o /dev/null
curl "${A[@]}" -X POST $R/branches -d '{"new_branch_name":"feature-conflict","old_branch_name":"main"}' -o /dev/null
curl "${A[@]}" -X PUT $R/contents/README.md -d "{\"content\":\"$(b64 'conflict side A')\",\"message\":\"Edit A\",\"branch\":\"feature-conflict\",\"sha\":\"$(sha feature-conflict)\"}" -o /dev/null
curl "${A[@]}" -X PUT $R/contents/README.md -d "{\"content\":\"$(b64 'conflict side B')\",\"message\":\"Edit B\",\"branch\":\"main\",\"sha\":\"$(sha main)\"}" -o /dev/null
curl "${A[@]}" -X POST $R/pulls -d '{"title":"Add launch brief","head":"feature-brief","base":"main"}' -o /dev/null
curl "${A[@]}" -X POST $R/pulls -d '{"title":"Rewrite README","head":"feature-conflict","base":"main"}' -o /dev/null
curl "${A[@]}" -X POST $R/branch_protections -d '{"rule_name":"main","required_approvals":1}' -o /dev/null
echo "Gitea seeded at http://127.0.0.1:$PORT (owner/viewer/maintainer)"
