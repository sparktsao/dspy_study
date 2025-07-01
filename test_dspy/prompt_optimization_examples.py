import dspy
from typing import List, Dict, Any

# Example 1: Basic Prompt Optimization (Fine-tuning the prompt, not the model)
def example_basic_prompt_optimization():
    """
    Example of prompt optimization - this fine-tunes the PROMPT, not the model weights.
    The model stays the same, but we find better ways to prompt it.
    """
    
    # Configure base model (model weights remain unchanged)
    # Using OpenAI-compatible API
    LLM_API_ENDPOINT = "your-api-endpoint-here"  # Replace with your endpoint
    LLM_API_KEY = "your-api-key-here"           # Replace with your key
    
    lm = dspy.LM(
        model="gpt-4o-mini",                # Your model name
        api_base=LLM_API_ENDPOINT,          # Your OpenAI-compatible endpoint
        api_key=LLM_API_KEY,                # Your API key
        temperature=0.0,
        max_tokens=16384
    )
    dspy.settings.configure(lm=lm)
    
    # Define signature for field mapping
    class FieldMapping(dspy.Signature):
        """Map Microsoft Defender fields to Trend Micro Common Schema fields."""
        defender_field = dspy.InputField(desc="Microsoft Defender field name")
        trend_micro_field = dspy.OutputField(desc="Corresponding Trend Micro field")
    
    # Create basic predictor
    field_mapper = dspy.Predict(FieldMapping)
    
    # Test before optimization
    print("=== Before Prompt Optimization ===")
    result_before = field_mapper(defender_field="SHA256")
    print(f"Input: SHA256")
    print(f"Output: {result_before.trend_micro_field}")
    print()
    
    # Create training examples for prompt optimization
    trainset = [
        dspy.Example(
            defender_field="SHA256",
            trend_micro_field="objectFileHashSha256"
        ).with_inputs('defender_field'),
        dspy.Example(
            defender_field="ProcessName", 
            trend_micro_field="objectProcessName"
        ).with_inputs('defender_field'),
        dspy.Example(
            defender_field="IPAddress",
            trend_micro_field="objectIpAddress"
        ).with_inputs('defender_field'),
        dspy.Example(
            defender_field="MD5",
            trend_micro_field="objectFileHashMd5"
        ).with_inputs('defender_field'),
        dspy.Example(
            defender_field="FileName",
            trend_micro_field="objectFileName"
        ).with_inputs('defender_field')
    ]
    
    # Define evaluation metric
    def exact_match_metric(example, pred, trace=None):
        return example.trend_micro_field.lower() == pred.trend_micro_field.lower()
    
    # Optimize the prompt using BootstrapFewShot
    optimizer = dspy.BootstrapFewShot(
        metric=exact_match_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2
    )
    
    # Compile (this optimizes the prompt, not the model)
    optimized_mapper = optimizer.compile(field_mapper, trainset=trainset)
    
    # Test after optimization
    print("=== After Prompt Optimization ===")
    result_after = optimized_mapper(defender_field="SHA256")
    print(f"Input: SHA256")
    print(f"Output: {result_after.trend_micro_field}")
    print()
    
    # Show what changed (the prompt structure)
    print("=== What Changed ===")
    print("The model is the same, but now it has:")
    print("- Better examples in the prompt")
    print("- Optimized prompt structure")
    print("- Few-shot examples that guide better responses")
    
    return optimized_mapper

# Example 2: Advanced Prompt Optimization with MIPROv2
def example_advanced_prompt_optimization():
    """
    Advanced prompt optimization using MIPROv2 - this is prompt fine-tuning.
    """
    
    # Using OpenAI-compatible API
    LLM_API_ENDPOINT = "your-api-endpoint-here"  # Replace with your endpoint
    LLM_API_KEY = "your-api-key-here"           # Replace with your key
    
    lm = dspy.LM(
        model="gpt-4o-mini",                # Your model name
        api_base=LLM_API_ENDPOINT,          # Your OpenAI-compatible endpoint
        api_key=LLM_API_KEY,                # Your API key
        temperature=0.0,
        max_tokens=16384
    )
    dspy.settings.configure(lm=lm)
    
    # More complex signature with reasoning
    class FieldMappingWithReasoning(dspy.Signature):
        """Map Microsoft Defender fields to Trend Micro Common Schema with reasoning."""
        defender_field = dspy.InputField(desc="Microsoft Defender field name")
        context = dspy.InputField(desc="Context about the field usage")
        reasoning = dspy.OutputField(desc="Reasoning for the mapping")
        trend_micro_field = dspy.OutputField(desc="Trend Micro field name")
    
    # Use Chain of Thought for better reasoning
    field_mapper = dspy.ChainOfThought(FieldMappingWithReasoning)
    
    # Extended training set
    trainset = [
        dspy.Example(
            defender_field="SHA256",
            context="File hash for malware detection",
            reasoning="SHA256 is a file hash, so it maps to the object file hash field",
            trend_micro_field="objectFileHashSha256"
        ).with_inputs('defender_field', 'context'),
        dspy.Example(
            defender_field="ProcessName",
            context="Name of running process",
            reasoning="Process name refers to the executable name, mapping to process object",
            trend_micro_field="objectProcessName"
        ).with_inputs('defender_field', 'context'),
        dspy.Example(
            defender_field="IPAddress",
            context="Network connection endpoint",
            reasoning="IP address is a network identifier, maps to IP object field",
            trend_micro_field="objectIpAddress"
        ).with_inputs('defender_field', 'context'),
        dspy.Example(
            defender_field="CommandLine",
            context="Process execution command",
            reasoning="Command line shows how process was executed, maps to command object",
            trend_micro_field="objectCommandLine"
        ).with_inputs('defender_field', 'context')
    ]
    
    # Evaluation metric that considers both reasoning and result
    def comprehensive_metric(example, pred, trace=None):
        field_match = example.trend_micro_field.lower() == pred.trend_micro_field.lower()
        reasoning_quality = len(pred.reasoning.split()) > 5  # Basic reasoning check
        return field_match and reasoning_quality
    
    # Use MIPROv2 for advanced prompt optimization
    optimizer = dspy.MIPROv2(
        metric=comprehensive_metric,
        auto="light",  # Can be "light", "medium", or "heavy"
        num_threads=4
    )
    
    print("=== Starting Advanced Prompt Optimization ===")
    print("This will try different prompt variations and select the best ones...")
    
    # Optimize the prompt
    optimized_mapper = optimizer.compile(field_mapper, trainset=trainset)
    
    # Test the optimized prompt
    result = optimized_mapper(
        defender_field="RegistryKey",
        context="Windows registry modification"
    )
    
    print(f"Field: {result.trend_micro_field}")
    print(f"Reasoning: {result.reasoning}")
    
    return optimized_mapper

# Example 3: Comparing Different Prompt Optimization Techniques
def compare_prompt_optimization_techniques():
    """
    Compare different prompt optimization (prompt fine-tuning) techniques.
    """
    
    # Using OpenAI-compatible API
    LLM_API_ENDPOINT = "your-api-endpoint-here"  # Replace with your endpoint
    LLM_API_KEY = "your-api-key-here"           # Replace with your key
    
    lm = dspy.LM(
        model="gpt-4o-mini",                # Your model name
        api_base=LLM_API_ENDPOINT,          # Your OpenAI-compatible endpoint
        api_key=LLM_API_KEY,                # Your API key
        temperature=0.0,
        max_tokens=16384
    )
    dspy.settings.configure(lm=lm)
    
    class SimpleMapping(dspy.Signature):
        """Map fields between security schemas."""
        input_field = dspy.InputField()
        output_field = dspy.OutputField()
    
    # Base predictor
    base_predictor = dspy.Predict(SimpleMapping)
    
    # Training data
    trainset = [
        dspy.Example(input_field="SHA256", output_field="objectFileHashSha256").with_inputs('input_field'),
        dspy.Example(input_field="ProcessName", output_field="objectProcessName").with_inputs('input_field'),
        dspy.Example(input_field="IPAddress", output_field="objectIpAddress").with_inputs('input_field'),
    ]
    
    def simple_metric(example, pred, trace=None):
        return example.output_field.lower() == pred.output_field.lower()
    
    print("=== Comparing Prompt Optimization Techniques ===\n")
    
    # 1. BootstrapFewShot - adds good examples to prompts
    print("1. BootstrapFewShot (adds examples to prompt):")
    bootstrap_optimizer = dspy.BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=2)
    bootstrap_optimized = bootstrap_optimizer.compile(base_predictor, trainset=trainset)
    
    result1 = bootstrap_optimized(input_field="MD5")
    print(f"   Result: {result1.output_field}")
    
    # 2. LabeledFewShot - uses provided examples directly
    print("\n2. LabeledFewShot (uses labeled examples):")
    labeled_optimizer = dspy.LabeledFewShot(k=2)  # Use 2 examples
    labeled_optimized = labeled_optimizer.compile(base_predictor, trainset=trainset)
    
    result2 = labeled_optimized(input_field="MD5")
    print(f"   Result: {result2.output_field}")
    
    # 3. MIPROv2 - advanced prompt optimization
    print("\n3. MIPROv2 (advanced prompt optimization):")
    mipro_optimizer = dspy.MIPROv2(metric=simple_metric, auto="light", num_threads=2)
    mipro_optimized = mipro_optimizer.compile(base_predictor, trainset=trainset)
    
    result3 = mipro_optimized(input_field="MD5")
    print(f"   Result: {result3.output_field}")
    
    print("\n=== What Each Technique Does ===")
    print("BootstrapFewShot: Finds good examples and adds them to prompts")
    print("LabeledFewShot: Uses your provided examples directly in prompts") 
    print("MIPROv2: Tries many prompt variations and picks the best performing ones")
    print("\nAll techniques optimize the PROMPT, not the model weights!")

# Example 4: ReAct Prompt Optimization (like your HotPotQA example)
def example_react_prompt_optimization():
    """
    Example similar to your HotPotQA code - this optimizes ReAct prompts, not model weights.
    """
    
    # Using OpenAI-compatible API
    LLM_API_ENDPOINT = "your-api-endpoint-here"  # Replace with your endpoint
    LLM_API_KEY = "your-api-key-here"           # Replace with your key
    
    lm = dspy.LM(
        model="gpt-4o-mini",                # Your model name
        api_base=LLM_API_ENDPOINT,          # Your OpenAI-compatible endpoint
        api_key=LLM_API_KEY,                # Your API key
        temperature=0.0,
        max_tokens=16384
    )
    dspy.settings.configure(lm=lm)
    
    # Mock search function (replace with real search)
    def search_field_mappings(query: str) -> list[str]:
        """Mock search function for field mapping knowledge base."""
        mappings = {
            "sha256": "objectFileHashSha256 - Used for file hash identification",
            "process": "objectProcessName - Used for process identification", 
            "ip": "objectIpAddress - Used for network connection tracking",
            "file": "objectFileName - Used for file identification"
        }
        
        results = []
        for key, value in mappings.items():
            if key.lower() in query.lower():
                results.append(value)
        
        return results[:3]  # Return top 3 results
    
    # Create ReAct module for field mapping
    react_mapper = dspy.ReAct(
        "question -> answer", 
        tools=[search_field_mappings]
    )
    
    # Training examples for prompt optimization
    trainset = [
        dspy.Example(
            question="What Trend Micro field maps to Microsoft Defender SHA256?",
            answer="objectFileHashSha256"
        ).with_inputs('question'),
        dspy.Example(
            question="What field should I use for process names in Trend Micro?",
            answer="objectProcessName"
        ).with_inputs('question'),
        dspy.Example(
            question="How do I map IP addresses to Trend Micro schema?",
            answer="objectIpAddress"
        ).with_inputs('question')
    ]
    
    # Evaluation metric
    def field_mapping_accuracy(example, pred, trace=None):
        expected = example.answer.lower()
        actual = pred.answer.lower()
        return expected in actual or actual in expected
    
    print("=== ReAct Prompt Optimization (like HotPotQA) ===")
    print("This optimizes how ReAct structures its reasoning and tool usage...")
    
    # Optimize the ReAct prompts
    optimizer = dspy.MIPROv2(
        metric=field_mapping_accuracy,
        auto="light",
        num_threads=4
    )
    
    optimized_react = optimizer.compile(react_mapper, trainset=trainset)
    
    # Test the optimized ReAct
    result = optimized_react(question="What Trend Micro field maps to Microsoft Defender file hash?")
    print(f"Question: What Trend Micro field maps to Microsoft Defender file hash?")
    print(f"Answer: {result.answer}")
    
    print("\n=== What Got Optimized ===")
    print("✓ ReAct reasoning patterns")
    print("✓ Tool usage strategies") 
    print("✓ Prompt structure for better tool integration")
    print("✓ Few-shot examples for better performance")
    print("✗ Model weights (unchanged)")
    
    return optimized_react

# Example 5: Inspecting Optimized Prompts
def inspect_optimized_prompts():
    """
    Show how to inspect what the prompt optimization actually changed.
    """
    
    # Using OpenAI-compatible API
    LLM_API_ENDPOINT = "your-api-endpoint-here"  # Replace with your endpoint
    LLM_API_KEY = "your-api-key-here"           # Replace with your key
    
    lm = dspy.LM(
        model="gpt-4o-mini",                # Your model name
        api_base=LLM_API_ENDPOINT,          # Your OpenAI-compatible endpoint
        api_key=LLM_API_KEY,                # Your API key
        temperature=0.0,
        max_tokens=16384
    )
    dspy.settings.configure(lm=lm)
    
    class FieldMapping(dspy.Signature):
        """Map security fields between schemas."""
        input_field = dspy.InputField()
        output_field = dspy.OutputField()
    
    # Original predictor
    original_predictor = dspy.Predict(FieldMapping)
    
    # Training data
    trainset = [
        dspy.Example(input_field="SHA256", output_field="objectFileHashSha256").with_inputs('input_field'),
        dspy.Example(input_field="ProcessName", output_field="objectProcessName").with_inputs('input_field'),
    ]
    
    # Optimize
    optimizer = dspy.BootstrapFewShot(
        metric=lambda ex, pred, trace: ex.output_field == pred.output_field,
        max_bootstrapped_demos=2
    )
    
    optimized_predictor = optimizer.compile(original_predictor, trainset=trainset)
    
    print("=== Inspecting Prompt Changes ===")
    
    # Show the difference
    print("ORIGINAL predictor:")
    print("- Uses basic signature prompt")
    print("- No examples in prompt")
    print("- Relies on model's base knowledge")
    
    print("\nOPTIMIZED predictor:")
    print("- Same model, enhanced prompt")
    print("- Includes few-shot examples")
    print("- Better structured instructions")
    
    # Test both
    test_input = "MD5"
    
    print(f"\nTesting with input: {test_input}")
    
    original_result = original_predictor(input_field=test_input)
    print(f"Original: {original_result.output_field}")
    
    optimized_result = optimized_predictor(input_field=test_input)
    print(f"Optimized: {optimized_result.output_field}")
    
    return original_predictor, optimized_predictor

# Example 6: Your HotPotQA Example Explained
def explain_hotpotqa_example():
    """
    Explanation of your HotPotQA example - this is PROMPT optimization, not model fine-tuning.
    """
    
    print("=== Your HotPotQA Example Analysis ===")
    print()
    print("Your code:")
    print("```python")
    print("trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]")
    print("react = dspy.ReAct('question -> answer', tools=[search_wikipedia])")
    print("tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto='light', num_threads=24)")
    print("optimized_react = tp.compile(react, trainset=trainset)")
    print("```")
    print()
    print("=== What This Does (PROMPT Fine-tuning) ===")
    print("✓ Takes your ReAct module (reasoning + tool usage)")
    print("✓ Tests it on 500 HotPotQA training examples")
    print("✓ Tries different prompt structures and few-shot examples")
    print("✓ Measures performance using answer_exact_match")
    print("✓ Selects the best-performing prompt configuration")
    print("✓ Returns an optimized ReAct with better prompts")
    print()
    print("=== What This Does NOT Do ===")
    print("✗ Does NOT modify gpt-4o-mini's weights")
    print("✗ Does NOT create a new fine-tuned model")
    print("✗ Does NOT train the underlying neural network")
    print()
    print("=== The Result ===")
    print("- Same model (gpt-4o-mini)")
    print("- Better prompts for reasoning")
    print("- Better examples for tool usage")
    print("- Improved question-answering performance")
    print()
    print("This is called 'prompt optimization' or 'prompt fine-tuning'")
    print("The 'fine-tuning' refers to fine-tuning the PROMPT, not the MODEL")

if __name__ == "__main__":
    print("DSPy Prompt Optimization Examples")
    print("=" * 50)
    print("These examples show PROMPT fine-tuning, not MODEL fine-tuning")
    print()
    
    # Uncomment the example you want to run:
    # example_basic_prompt_optimization()
    # example_advanced_prompt_optimization()
    # compare_prompt_optimization_techniques()
    # example_react_prompt_optimization()
    # inspect_optimized_prompts()
    explain_hotpotqa_example()
