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

def print_prompt_messages(messages, title):
    """Helper function to print messages in a readable format"""
    print(f"=== {title} ===")
    for i, msg in enumerate(messages):
        role = msg['role'].upper()
        content = msg['content']
        print(f"[{role}]: {content}")
        if i < len(messages) - 1:  # Don't print separator after last message
            print("-" * 40)
    print("=" * 80)
    print()

# 6. Test BEFORE optimization and capture prompt
print("Testing BEFORE optimization...")
lm.history.clear()  # Clear history
result_before = predictor(defender_field="CommandLine")

# Get the original prompt from history
if lm.history:
    original_messages = lm.history[-1]['messages']
    print_prompt_messages(original_messages, "ORIGINAL PROMPT (Before Optimization)")
    print(f"Result: {result_before.trend_micro_field}")
    print()

# 7. Optimize the prompt
print("=== OPTIMIZING PROMPT ===")
optimizer = dspy.BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=3)
optimized_predictor = optimizer.compile(predictor, trainset=trainset)
print("Optimization complete!")
print()

# 8. Test AFTER optimization and capture prompt
print("Testing AFTER optimization...")
lm.history.clear()  # Clear history to get clean optimized prompt
result_after = optimized_predictor(defender_field="CommandLine")

# Get the optimized prompt from history
if lm.history:
    optimized_messages = lm.history[-1]['messages']
    print_prompt_messages(optimized_messages, "OPTIMIZED PROMPT (After Optimization)")
    print(f"Result: {result_after.trend_micro_field}")
    print()

# 9. Show the key differences
print("=== ANALYSIS ===")
print("BEFORE OPTIMIZATION:")
print("- Simple system message with field descriptions")
print("- Single user message with the input")
print("- No examples provided")
print("- Model relies on its base knowledge")
print()

print("AFTER OPTIMIZATION:")
print("- Same system message structure")
print("- Multiple conversation examples (few-shot learning)")
print("- Shows input-output patterns from training data")
print("- Guides model with concrete examples")
print("- Uses special markers for structured output")
print()

print("KEY INSIGHT:")
print("DSPy automatically selected the best examples from your training data")
print("and formatted them as a conversation to guide the model!")
