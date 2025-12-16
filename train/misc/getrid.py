python - <<'PY'
import os, sys, textwrap
from huggingface_hub import HfApi, whoami, create_repo, upload_folder

token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
lora_dir = os.environ.get("LORA_DIR")

if not lora_dir or not os.path.isdir(lora_dir):
    print(f"[ERROR] LORA_DIR missing or not a directory: {lora_dir}", file=sys.stderr)
    sys.exit(1)

api = HfApi()
user = whoami(token=token)["name"]
repo_id = f"{user}/arc-lora-llama3-3b"   # change name if you want

# Create (or reuse) a private model repo
create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=token)

# Minimal README so the repo is self-explanatory
base_model = "chuanli11/Llama-3.2-3B-Instruct-uncensored"
readme = textwrap.dedent(f"""\
# ARC LoRA Adapter for Llama 3.2 3B
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- qlora
- causal-lm
- arc

This repo contains the **LoRA adapter only**.

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "{base_model}"
adapter = "{repo_id}"

tok = AutoTokenizer.from_pretrained(base)
base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter)
""")
with open(os.path.join(lora_dir, "README.md"), "w") as f:
    f.write(readme)

print(f"[INFO] Uploading {lora_dir} -> {repo_id} ...")
upload_folder(
folder_path=lora_dir,
repo_id=repo_id,
repo_type="model",
token=token,
commit_message="Initial upload of LoRA adapter",
ignore_patterns=["/runs/","/logs/","/*.pt","/tmp_*"],
)
print(f"[DONE] View your repo at: https://huggingface.co/{repo_id}")
PY