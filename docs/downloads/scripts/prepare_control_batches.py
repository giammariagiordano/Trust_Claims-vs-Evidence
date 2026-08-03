import json
import os

IN_FILE = "control_raw.jsonl"
BATCH_DIR = "control_batches"
N_BATCHES = 6
MAX_CARD_CHARS = 3000

os.makedirs(BATCH_DIR, exist_ok=True)

records = []
with open(IN_FILE) as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} records")

with open("control_metadata.jsonl", "w") as f:
    for r in records:
        meta = {k: v for k, v in r.items() if k != "card_text"}
        f.write(json.dumps(meta) + "\n")

batch_size = (len(records) + N_BATCHES - 1) // N_BATCHES
for i in range(N_BATCHES):
    chunk = records[i * batch_size: (i + 1) * batch_size]
    if not chunk:
        continue
    out = [{"id": r["id"], "card_text": r["card_text"][:MAX_CARD_CHARS]} for r in chunk]
    with open(f"{BATCH_DIR}/cbatch_{i:02d}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"cbatch_{i:02d}.json: {len(out)} models")
