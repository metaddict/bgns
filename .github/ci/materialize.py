from __future__ import annotations

import base64
import shutil
import tarfile
from pathlib import Path

repo = Path.cwd()
payload = repo / ".github" / "ci" / "bgns_0.4.2.tar.gz.b64"
archive = repo / ".github" / "ci" / "bgns_0.4.2.input.tar.gz"
staging = repo / ".github" / "ci" / "materialized"

archive.write_bytes(base64.b64decode(payload.read_text(encoding="utf-8")))
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)

with tarfile.open(archive, "r:gz") as tf:
    members = tf.getmembers()
    root = staging.resolve()
    for member in members:
        target = (staging / member.name).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
    tf.extractall(staging)

src = staging / "bgns"
if not (src / "DESCRIPTION").is_file():
    raise RuntimeError("The payload does not contain bgns/DESCRIPTION")

for item in src.iterdir():
    target = repo / item.name
    if item.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(item, target)
    else:
        shutil.copy2(item, target)

print((repo / "DESCRIPTION").read_text(encoding="utf-8"))
