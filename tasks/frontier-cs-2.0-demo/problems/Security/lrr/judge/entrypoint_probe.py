"""Execute the retained prompt-construction entrypoint without loading a model."""
import os
from pathlib import Path
import runpy
import sys
import types

task_dir = Path(__file__).resolve().parent
repo = task_dir / "harbor" / "app"
os.chdir(repo)
sys.path.insert(0, str(repo / "scripts"))
# inference.py imports its training stack eagerly although the retained prompt
# constructor does not use it. Match the evaluator's CPU import adaptation.
peft = types.ModuleType("peft")
peft.PeftModel = object
transformers = types.ModuleType("transformers")
transformers.AutoModelForCausalLM = object
transformers.AutoTokenizer = object
transformers.BitsAndBytesConfig = object
tqdm = types.ModuleType("tqdm")
tqdm.tqdm = lambda value, *args, **kwargs: value
sys.modules.update(peft=peft, transformers=transformers, tqdm=tqdm)
if Path("/app/scripts/inference.py").is_file():
    namespace = runpy.run_path("/app/scripts/inference.py", run_name="lrr_authentic_inference")
else:
    namespace = runpy.run_path("scripts/inference.py", run_name="lrr_authentic_inference")
if "get_line_to_chat_and_pattern" not in namespace:
    raise SystemExit("authentic constructor is missing")
