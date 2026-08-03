from huggingface_hub import HfApi

api = HfApi()
tags = ["medical", "clinical", "healthcare", "biomedical", "radiology", "pathology"]

seen = {}
total_ids = set()
for tag in tags:
    models = list(api.list_models(filter=tag, limit=None))
    seen[tag] = len(models)
    for m in models:
        total_ids.add(m.id)

for tag, n in seen.items():
    print(f"tag={tag}: {n} models")

print(f"UNION (dedup) across tags: {len(total_ids)} models")
