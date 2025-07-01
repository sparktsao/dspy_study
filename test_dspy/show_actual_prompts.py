import dspy

# 1. Setup your OpenAI-compatible API
LLM_API_ENDPOINT = "your-api-endpoint-here"
LLM_API_KEY = "your-api-key-here"

lm = dspy.LM(
    model="gpt-4o-mini",
    api_base=LLM_API_ENDPOINT,
    api_key=LLM_API_KEY,
    temperature=0.0,
    max_tokens=16384
)
dspy.settings.configure(lm=lm)

# 2. Define your task signature
class FieldMapping(dspy.Signature):
    """Map Microsoft Defender fields to Trend Micro fields."""
    defender_field = dspy.InputField()
    trend_micro_field = dspy.OutputField()

# 3. Create predictor
predictor = dspy.Predict(FieldMapping)

# 4. Create your dataset
trainset = [
    dspy.Example(defender_field="SHA256", trend_micro_field="objectFileHashSha256").with_inputs('defender_field'),
    dspy.Example(defender_field="ProcessName", trend_micro_field="objectProcessName").with_inputs('defender_field'),
    dspy.Example(defender_field="IPAddress", trend_micro_field="objectIpAddress").with_inputs('defender_field'),
    dspy.Example(defender_field="MD5", trend_micro_field="objectFileHashMd5").with_inputs('defender_field'),
    dspy.Example(defender_field="FileName", trend_micro_field="objectFileName").with_inputs('defender_field')
]

# 5. Define evaluation metric
def exact_match(example, pred, trace=None):
    return example.trend_micro_field.lower() == pred.trend_micro_field.lower()

# 6. Capture the ORIGINAL prompt by intercepting the LM call
print("=== CAPTURING ORIGINAL PROMPT ===")

# Store the original completion function
original_call = lm.__call__

# Create a wrapper to capture prompts
captured_prompts = []

def capture_prompt_wrapper(prompt=None, messages=None, **kwargs):
    if messages:
        # For chat format, capture the full conversation
        captured_prompts.append({
            "type": "messages",
            "content": messages
        })
    else:
        # For text format, capture the prompt
        captured_prompts.append({
            "type": "prompt", 
            "content": prompt
        })
    return original_call(prompt=prompt, messages=messages, **kwargs)

# Replace the LM call with our wrapper
lm.__call__ = capture_prompt_wrapper

# Test the original predictor to capture its prompt
result_before = predictor(defender_field="CommandLine")

# Show the original prompt
print("ORIGINAL PROMPT:")
if captured_prompts:
    original_prompt = captured_prompts[-1]
    if original_prompt["type"] == "messages":
        for msg in original_prompt["content"]:
            print(f"[{msg['role']}]: {msg['content']}")
    else:
        print(original_prompt["content"])
print()
print(f"Original Result: {result_before.trend_micro_field}")
print("=" * 80)
print()

# 7. Optimize the prompt
print("=== OPTIMIZING PROMPT ===")
optimizer = dspy.BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=3)
optimized_predictor = optimizer.compile(predictor, trainset=trainset)
print("Optimization complete!")
print()

# 8. Clear captured prompts and test optimized version
captured_prompts.clear()
result_after = optimized_predictor(defender_field="CommandLine")

# Show the optimized prompt
print("=== OPTIMIZED PROMPT ===")
if captured_prompts:
    optimized_prompt = captured_prompts[-1]
    if optimized_prompt["type"] == "messages":
        for msg in optimized_prompt["content"]:
            print(f"[{msg['role']}]: {msg['content']}")
    else:
        print(optimized_prompt["content"])
print()
print(f"Optimized Result: {result_after.trend_micro_field}")
print("=" * 80)
print()

# 9. Show the key differences
print("=== KEY DIFFERENCES ===")
print("ORIGINAL PROMPT:")
print("- Basic task description only")
print("- No examples provided")
print("- Minimal guidance")
print()
print("OPTIMIZED PROMPT:")
print("- Same task description")
print("- Includes few-shot examples")
print("- Examples show input-output patterns")
print("- Guides the model to correct format")
print()

# Restore original LM function
lm.__call__ = original_call

# 10. Alternative method using DSPy's built-in inspection
print("=== ALTERNATIVE: Using DSPy History ===")
print("You can also inspect prompts through lm.history:")
if lm.history:
    last_call = lm.history[-1]
    print("Last prompt messages:")
    for msg in last_call.get('messages', []):
        print(f"[{msg['role']}]: {msg['content'][:200]}...")
