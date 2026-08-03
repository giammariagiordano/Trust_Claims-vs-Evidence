import json
import random
import time
import requests
from huggingface_hub import HfApi

api = HfApi()
TAGS = ["medical", "clinical", "healthcare", "biomedical", "radiology", "pathology"]
TARGET_N = 500
SEED = 42
OUT_FILE = "models_raw.jsonl"

random.seed(SEED)


def collect_candidates():
    by_id = {}
    for tag in TAGS:
        for m in api.list_models(filter=tag, full=True, cardData=True, limit=None):
            by_id[m.id] = m  # dedup by model id
    return list(by_id.values())


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
    print("Collecting candidate models from tags...")
    candidates = collect_candidates()
    print(f"Total unique candidates: {len(candidates)}")

    random.shuffle(candidates)

    org_cache = {}
    results = []
    tried = 0

    for m in candidates:
        if len(results) >= TARGET_N:
            break
        tried += 1

        card_text = fetch_card_text(m.id)
        if not card_text or len(card_text.strip()) < 50:
            continue  # skip empty/near-empty cards, don't count toward target

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
            print(f"  collected {len(results)}/{TARGET_N} (tried {tried}/{len(candidates)})")

    print(f"Done. Collected {len(results)} valid models out of {tried} tried "
          f"({len(candidates)} total candidates available).")

    with open(OUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
