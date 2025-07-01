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

# 6. Test BEFORE optimization
print("=== BEFORE PROMPT OPTIMIZATION ===")
result_before = predictor(defender_field="CommandLine")
print(f"Input: CommandLine")
print(f"Output: {result_before.trend_micro_field}")
print()

# Show what the original prompt looks like
print("Original prompt structure:")
print("- Basic signature description")
print("- No examples included")
print("- Relies on model's base knowledge")
print()

# 7. Optimize the prompt (this is the "fine-tuning")
print("=== OPTIMIZING PROMPT ===")
print("Finding best examples to include in prompt...")
optimizer = dspy.BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=3)
optimized_predictor = optimizer.compile(predictor, trainset=trainset)
print("Optimization complete!")
print()

# 8. Test AFTER optimization
print("=== AFTER PROMPT OPTIMIZATION ===")
result_after = optimized_predictor(defender_field="CommandLine")
print(f"Input: CommandLine")
print(f"Output: {result_after.trend_micro_field}")
print()

# Show what changed
print("Optimized prompt now includes:")
print("- Same signature description")
print("- Few-shot examples automatically selected")
print("- Better structured prompts")
print("- Examples that guide the model to correct answers")
print()

# 9. Show the difference with another test
print("=== COMPARISON TEST ===")
test_input = "RegistryKey"

original_result = predictor(defender_field=test_input)
optimized_result = optimized_predictor(defender_field=test_input)

print(f"Input: {test_input}")
print(f"Original output: {original_result.trend_micro_field}")
print(f"Optimized output: {optimized_result.trend_micro_field}")
print()

print("The model is the same, but the PROMPT is now optimized!")
