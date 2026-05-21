#!/usr/bin/env bash
# Pull all files from the agent's HF Space sandbox /app/models into ./oculomics-output/models.
# Usage: ./sync-sandbox.sh [<sandbox-slug>]
# Default slug is read from the latest sandbox_create result in any active session.
set -e
SLUG="${1:-}"
if [ -z "$SLUG" ]; then
  SLUG=$(curl -s http://localhost:7860/api/sessions | python3 -c "
import sys, json, urllib.request, re
sessions = json.load(sys.stdin)
for s in sessions:
    sid = s['session_id']
    msgs = json.loads(urllib.request.urlopen(f'http://localhost:7860/api/session/{sid}/messages').read())
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get('name') == 'sandbox_create':
            c = m.get('content','') or ''
            if isinstance(c, list): c = str(c)
            mt = re.search(r'([\\w-]+/sandbox-[a-f0-9]+)', c)
            if mt:
                print(mt.group(1).lower().replace('/','-'))
                exit
")
fi
[ -z "$SLUG" ] && echo "no sandbox slug found" && exit 1
echo "syncing sandbox: $SLUG"
# Push from sandbox → HF repo, then pull repo → host (bypasses the bash output size limit)
curl -sX POST "https://$SLUG.hf.space/api/bash" \
  -H 'content-type: application/json' \
  -d '{"command": "python3 -c \"import os; from huggingface_hub import upload_folder, login; login(token=os.environ[\\\"HF_TOKEN\\\"]); upload_folder(repo_id=\\\"Mazhar004/oculomics-proxy\\\", folder_path=\\\"/app/models\\\", path_in_repo=\\\"models\\\", commit_message=\\\"sync sandbox artifacts\\\")\"", "cwd": "/app", "timeout": 300}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('upload:', 'OK' if d.get('success') else d.get('error',''))"
docker compose run --rm oculomics python -c "
from huggingface_hub import snapshot_download
import os, shutil, pathlib
snapshot_download(repo_id='Mazhar004/oculomics-proxy', allow_patterns=['models/*'], local_dir='/workspace/.tmp-download', token=os.environ.get('HF_TOKEN'))
src = pathlib.Path('/workspace/.tmp-download/models'); dst = pathlib.Path('/workspace/models')
dst.mkdir(exist_ok=True)
for item in src.iterdir():
    target = dst / item.name
    if item.is_dir(): shutil.copytree(item, target, dirs_exist_ok=True)
    else: shutil.copy2(item, target)
shutil.rmtree('/workspace/.tmp-download')
print('downloaded')
" 2>&1 | tail -3
echo "files now on host:"
ls -lh oculomics-output/models/ | tail -10
