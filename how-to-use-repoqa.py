# %% [markdown]
# # Model Input/Output Inspector
# 
# This notebook allows you to inspect the input prompts and model outputs one by one.
# 

# %%
import json
import os
import torch
from repoqa.data import get_repoqa_data
from repoqa.search_needle_function import (
    make_code_context,
    make_task_id,
    INSTRUCTION,
    TEMPLATE,
    CleanComment
)
from repoqa.utility import topological_sort


# %% [markdown]
# ## Configuration
# 

# %%
# Configuration
# Set GPU device and environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizers warning

model = "Qwen/Qwen3-8B"  # Change this to your model
backend = "hf"  # Options: "openai", "vllm", "anthropic", "hf", "google"
code_context_size = 16 * 1024
max_new_tokens = 1024
temperature = 0.0
top_p = 1.0
top_k = None
system_message = None
languages = None  # None = all languages, or specify list like ["python"]
clean_ctx_comments = CleanComment.NoClean
trust_remote_code = False
attn_implementation = None
enable_thinking = None
torch_dtype = torch.bfloat16


# %% [markdown]
# ## Load Dataset and Prepare Tasks
# 

# %%
### this is full path 
### /mnt/shared-fs/gelvan/repoqa/cache_ntoken_16384_v1.jsonl

cache_file_path = f"cache_ntoken_{code_context_size}_v1.jsonl"

tasks = []
print(f"Loading tasks from {cache_file_path}...")
if os.path.exists(cache_file_path):
    with open(cache_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # The notebook is configured for python only.
            if data.get('language') == 'python':
                data.pop('cache_id', None)
                tasks.append(data)
    print(f"Loaded {len(tasks)} python tasks.")
else:
    print(f"Cache file not found: {cache_file_path}")

print(f"\nTotal tasks prepared: {len(tasks)}")

# %%


# %% [markdown]
# ## Initialize Model Backend
# 

# %%
'''os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Initialize the backend
try:
    print(f"Loading model: {model}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    from repoqa.provider.hf import HfProvider
    engine = HfProvider(
        model,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        enable_thinking=enable_thinking,
        torch_dtype=torch_dtype,
    )
    print(f"✅ Backend '{backend}' initialized successfully!")
except RuntimeError as e:
    print(f"❌ RuntimeError during initialization: {e}")
    print("\nPossible solutions:")
    print("1. Reduce model size or use a quantized version")
    print("2. Free up GPU memory by closing other processes")
    print("3. Use device_map='auto' for automatic memory management")
    raise
except Exception as e:
    print(f"❌ Error during initialization: {type(e).__name__}: {e}")
    raise
'''

# %%
# Process tasks one by one
for idx, task in enumerate(tasks[:5]):
    print("=" * 100)
    print(f"TASK {idx + 1}/{len(tasks)}")
    print("=" * 100)
    
    # Calculate actual position ratio
    actual_position_ratio = task["needle_token_start"] / task["code_context_ntokens"]
    
    print(f"\n📍 Task Info:")
    print(f"   - Name: {task['name']}")
    print(f"   - Repo: {task['repo']}")
    print(f"   - Language: {task['language']}")
    print(f"   - Position Ratio: actual={actual_position_ratio:.2f}, expected={task['position_ratio']:.2f}")
    print(f"   - Code Context Tokens: {task['code_context_ntokens']}")
    
    # Construct prompt from template
    prompt = ""
    for key in task["template"].split("\n"):
        print(key)
        print(task[key][:10])
        prompt += task[key]
    
    print(f"\n📥 INPUT (Prompt):")
    print("-" * 100)
    print(prompt[:3000])
    print("-" * 100)
    
    # Generate reply
    print(f"\n⏳ Generating reply...")
    '''replies = engine.generate_reply(
        prompt,
        n=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        system_msg=system_message
    )
    
    print(f"\n📤 OUTPUT (Model Reply):")
    print("-" * 100)
    for i, reply in enumerate(replies):
        if len(replies) > 1:
            print(f"\nReply {i+1}:")
        print(reply)
    print("-" * 100)'''
    
    print("\n" + "=" * 100 + "\n")


# %%



