import json
import random
import time
import requests
from huggingface_hub import HfApi

api = HfApi()
HEALTH_TAGS = {"medical", "clinical", "healthcare", "biomedical", "radiology", "pathology"}
ADDITIONAL_TARGET = 200
SEED = 44
OUT_FILE = "control_raw.jsonl"
POOL_LIMIT = 20000

random.seed(SEED)

existing_ids = set()
with open(OUT_FILE) as f:
    for line in f:
        existing_ids.add(json.loads(line)["id"])
print(f"Already have {len(existing_ids)} control models, need {ADDITIONAL_TARGET} more")


def is_health_related(tags, model_id):
    tagset = {t.lower() for t in (tags or [])}
    if tagset & HEALTH_TAGS:
        return True
    lower_id = model_id.lower()
    return any(h in lower_id for h in HEALTH_TAGS)


def is_organization(namespace):
    try:
        r = requests.get(f"https://huggingface.co/api/organizations/{namespace}/overview", timeout=10)
        return r.status_code == 200
    except requests.RequestException:
        return None


def fetch_card_text(model_id):
    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and len(r.text.strip()) > 0:
            return r.text
    except requests.RequestException:
        pass
    return None


def main():
    print(f"Pulling a broad pool of up to {POOL_LIMIT} recently-created models...")
    pool = list(api.list_models(sort="created_at", limit=POOL_LIMIT, full=True, cardData=False))
    print(f"Pool size: {len(pool)}")

    candidates = [
        m for m in pool
        if not is_health_related(getattr(m, "tags", None), m.id) and m.id not in existing_ids
    ]
    print(f"Non-health, not-already-collected candidates: {len(candidates)}")

    random.shuffle(candidates)

    org_cache = {}
    results = []
    tried = 0

    for m in candidates:
        if len(results) >= ADDITIONAL_TARGET:
            break
        tried += 1

        card_text = fetch_card_text(m.id)
        if not card_text or len(card_text.strip()) < 50:
            continue

        namespace = m.id.split("/")[0] if "/" in m.id else m.id
        if namespace not in org_cache:
            org_cache[namespace] = is_organization(namespace)
            time.sleep(0.05)
        is_org = org_cache[namespace]

        results.append({
            "id": m.id,
            "author": namespace,
            "is_organization": is_org,
            "downloads": getattr(m, "downloads", None),
            "likes": getattr(m, "likes", None),
            "created_at": str(getattr(m, "created_at", None)),
            "last_modified": str(getattr(m, "last_modified", None)),
            "pipeline_tag": getattr(m, "pipeline_tag", None),
            "tags": getattr(m, "tags", None),
            "card_text": card_text,
        })

        if len(results) % 25 == 0:
            print(f"  collected {len(results)}/{ADDITIONAL_TARGET} (tried {tried}/{len(candidates)})")

    print(f"Done. Collected {len(results)} new valid models out of {tried} tried.")

    with open(OUT_FILE, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Appended to {OUT_FILE}")


if __name__ == "__main__":
    main()
