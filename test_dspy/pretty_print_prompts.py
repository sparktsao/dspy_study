import dspy
from colorama import Fore, Back, Style
import colorama

# Initialize colorama for cross-platform color support
colorama.init()

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

def pretty_print_prompt(messages, title, color):
    """Pretty print prompt messages with colors"""
    print(f"\n{color}{'='*80}")
    print(f"{title}")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        if role == 'system':
            print(f"\n{Fore.CYAN}[SYSTEM]{Style.RESET_ALL}")
            print(f"{Fore.LIGHTCYAN_EX}{content}{Style.RESET_ALL}")
        elif role == 'user':
            print(f"\n{Fore.GREEN}[USER]{Style.RESET_ALL}")
            print(f"{Fore.LIGHTGREEN_EX}{content}{Style.RESET_ALL}")
        elif role == 'assistant':
            print(f"\n{Fore.YELLOW}[ASSISTANT]{Style.RESET_ALL}")
            print(f"{Fore.LIGHTYELLOW_EX}{content}{Style.RESET_ALL}")
    
    print(f"\n{color}{'='*80}{Style.RESET_ALL}\n")

# 6. Test BEFORE optimization
print(f"{Fore.MAGENTA}🔍 Testing BEFORE optimization...{Style.RESET_ALL}")
lm.history.clear()
result_before = predictor(defender_field="CommandLine")

if lm.history:
    original_messages = lm.history[-1]['messages']
    pretty_print_prompt(original_messages, "🔴 ORIGINAL PROMPT (Before Optimization)", Fore.RED)
    print(f"{Fore.BLUE}Result: {Style.BRIGHT}{result_before.trend_micro_field}{Style.RESET_ALL}\n")

print("--------------------------------------------------------")
# 7. Optimize the prompt
print(f"{Fore.MAGENTA}⚡ Optimizing prompt...{Style.RESET_ALL}")
optimizer = dspy.BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=3)
optimized_predictor = optimizer.compile(predictor, trainset=trainset)
print(f"{Fore.GREEN}✅ Optimization complete!{Style.RESET_ALL}\n")

# 8. Test AFTER optimization
print(f"{Fore.MAGENTA}🔍 Testing AFTER optimization...{Style.RESET_ALL}")
lm.history.clear()
result_after = optimized_predictor(defender_field="CommandLine")

if lm.history:
    optimized_messages = lm.history[-1]['messages']
    pretty_print_prompt(optimized_messages, "🟢 OPTIMIZED PROMPT (After Optimization)", Fore.GREEN)
    print(f"{Fore.BLUE}Result: {Style.BRIGHT}{result_after.trend_micro_field}{Style.RESET_ALL}\n")

# 9. Summary
print("--------------------------------------------------------")
print(f"{Fore.CYAN}{'='*80}")
print(f"📊 SUMMARY")
print(f"{'='*80}{Style.RESET_ALL}")
print(f"{Fore.RED}BEFORE:{Style.RESET_ALL} Simple prompt, no examples")
print(f"{Fore.GREEN}AFTER:{Style.RESET_ALL}  Rich prompt with few-shot examples")
print(f"{Fore.YELLOW}KEY:{Style.RESET_ALL}    Same model, better prompting strategy!")
print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
