import json
from pathlib import Path

BASE_PATH   = "log_20260422_221958_mode1_30.json"
SOURCE_PATH = "log_20260422_224621_mode2_30.json"
OUTPUT_PATH = "log_20260422_221958_mode1_30_augmented.json"

with open(BASE_PATH, 'r') as f:
    base_data = json.load(f)
with open(SOURCE_PATH, 'r') as f:
    source_data = json.load(f)

if not isinstance(base_data, list):
    base_data = [base_data]
if not isinstance(source_data, list):
    source_data = [source_data]

augmented = []
for session in source_data:
    messages = session.get("messages", {})
    ts = messages.get("2", [])
    if len(ts) >= 4 and all(ts[:4]):
        new_session = json.loads(json.dumps(session))
        new_session["root_anchor"] = 2
        new_session["messages"]["2"] = [1] + ts[:4]
        augmented.append(new_session)

result = base_data + augmented

with open(OUTPUT_PATH, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"base: {len(base_data)}개 + 추출: {len(augmented)}개 = 총 {len(result)}개")
print(f"저장 완료: {OUTPUT_PATH}")