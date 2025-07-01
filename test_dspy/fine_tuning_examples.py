import dspy
from typing import List, Dict, Any

# Example 1: Fine-tuning with DSPy (Experimental Feature)
def example_dspy_finetuning():
    """
    Example of actual model fine-tuning using DSPy's experimental fine-tuning feature.
    This modifies the model weights, unlike prompt optimization.
    """
    
    # Enable experimental features
    dspy.settings.experimental = True
    
    # Initialize a language model that supports fine-tuning
    lm = dspy.LM(
        model="openai/gpt-3.5-turbo",  # Base model to fine-tune
        api_key="your-api-key-here",
        finetuning_model="gpt-3.5-turbo",  # Model to use for fine-tuning
        temperature=0.0
    )
    
    dspy.settings.configure(lm=lm)
    
    # Prepare training data in the correct format
    # Each example should have input-output pairs
    train_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a cybersecurity expert mapping security fields."},
                {"role": "user", "content": "What Trend Micro field maps to Microsoft Defender SHA256?"},
                {"role": "assistant", "content": "objectFileHashSha256"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a cybersecurity expert mapping security fields."},
                {"role": "user", "content": "What Trend Micro field maps to Microsoft Defender process name?"},
                {"role": "assistant", "content": "objectProcessName"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a cybersecurity expert mapping security fields."},
                {"role": "user", "content": "What Trend Micro field maps to Microsoft Defender IP address?"},
                {"role": "assistant", "content": "objectIpAddress"}
            ]
        }
        # Add more training examples...
    ]
    
    # Start fine-tuning job
    try:
        training_job = lm.finetune(
            train_data=train_data,
            train_data_format="openai_chat",  # Format for chat-based fine-tuning
            train_kwargs={
                "n_epochs": 3,
                "batch_size": 1,
                "learning_rate_multiplier": 0.1
            }
        )
        
        print("Fine-tuning job started...")
        print(f"Job ID: {training_job}")
        
        # Wait for completion (this is a blocking operation)
        fine_tuned_lm = training_job.result()
        
        print("Fine-tuning completed!")
        print(f"Fine-tuned model: {fine_tuned_lm.model}")
        
        # Test the fine-tuned model
        response = fine_tuned_lm("What Trend Micro field maps to Microsoft Defender file hash?")
        print(f"Fine-tuned response: {response[0]}")
        
        return fine_tuned_lm
        
    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        return None

# Example 2: Prompt Optimization (NOT fine-tuning)
def example_prompt_optimization():
    """
    Example of prompt optimization using DSPy optimizers.
    This does NOT modify model weights - it optimizes prompting strategies.
    """
    
    # Configure base model
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key="your-api-key-here",
        temperature=0.0
    )
    dspy.settings.configure(lm=lm)
    
    # Define a signature for field mapping
    class FieldMapping(dspy.Signature):
        """Map Microsoft Defender fields to Trend Micro Common Schema fields."""
        defender_field = dspy.InputField(desc="Microsoft Defender field name")
        context = dspy.InputField(desc="Additional context about the field")
        trend_micro_field = dspy.OutputField(desc="Corresponding Trend Micro field")
    
    # Create a predictor
    field_mapper = dspy.Predict(FieldMapping)
    
    # Create training examples
    trainset = [
        dspy.Example(
            defender_field="SHA256",
            context="File hash for malware detection",
            trend_micro_field="objectFileHashSha256"
        ).with_inputs('defender_field', 'context'),
        dspy.Example(
            defender_field="ProcessName",
            context="Name of the running process",
            trend_micro_field="objectProcessName"
        ).with_inputs('defender_field', 'context'),
        dspy.Example(
            defender_field="IPAddress",
            context="Network connection IP address",
            trend_micro_field="objectIpAddress"
        ).with_inputs('defender_field', 'context')
        # Add more examples...
    ]
    
    # Define evaluation metric
    def field_mapping_metric(example, pred, trace=None):
        return example.trend_micro_field.lower() == pred.trend_micro_field.lower()
    
    # Optimize the prompt using MIPROv2
    optimizer = dspy.MIPROv2(
        metric=field_mapping_metric,
        auto="light",
        num_threads=4
    )
    
    # Compile (optimize) the predictor
    optimized_mapper = optimizer.compile(field_mapper, trainset=trainset)
    
    # Test the optimized predictor
    result = optimized_mapper(
        defender_field="SHA256",
        context="File hash for malware detection"
    )
    
    print(f"Optimized mapping result: {result.trend_micro_field}")
    
    return optimized_mapper

# Example 3: Creating training data for fine-tuning
def create_field_mapping_training_data():
    """
    Create properly formatted training data for field mapping fine-tuning.
    """
    
    # Field mapping examples
    field_mappings = [
        ("SHA256", "File hash", "objectFileHashSha256"),
        ("MD5", "File hash", "objectFileHashMd5"),
        ("ProcessName", "Process name", "objectProcessName"),
        ("ProcessId", "Process identifier", "objectProcessId"),
        ("IPAddress", "IP address", "objectIpAddress"),
        ("Port", "Network port", "objectPort"),
        ("URL", "Web URL", "objectUrl"),
        ("Domain", "Domain name", "objectDomain"),
        ("FileName", "File name", "objectFileName"),
        ("FilePath", "File path", "objectFilePath"),
        ("UserName", "User account", "objectUserName"),
        ("CommandLine", "Command line", "objectCommandLine"),
        ("RegistryKey", "Registry key", "objectRegistryKey"),
        ("EventId", "Event identifier", "objectEventId"),
        ("Timestamp", "Event timestamp", "objectTimestamp")
    ]
    
    training_data = []
    
    for defender_field, description, trend_micro_field in field_mappings:
        # Create multiple variations of the same mapping
        variations = [
            f"What Trend Micro field maps to Microsoft Defender {defender_field}?",
            f"Map Microsoft Defender {defender_field} to Trend Micro schema",
            f"What is the Trend Micro equivalent of {defender_field} from Microsoft Defender?",
            f"Convert Microsoft Defender {defender_field} ({description}) to Trend Micro field"
        ]
        
        for question in variations:
            training_data.append({
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a cybersecurity expert that maps Microsoft Defender fields to Trend Micro Common Schema fields. Provide only the exact field name."
                    },
                    {
                        "role": "user", 
                        "content": question
                    },
                    {
                        "role": "assistant", 
                        "content": trend_micro_field
                    }
                ]
            })
    
    return training_data

# Example 4: Complete fine-tuning workflow
def complete_finetuning_workflow():
    """
    Complete workflow for fine-tuning a model for field mapping.
    """
    
    print("=== DSPy Fine-tuning Workflow ===")
    
    # Step 1: Enable experimental features
    dspy.settings.experimental = True
    print("✓ Experimental features enabled")
    
    # Step 2: Create training data
    train_data = create_field_mapping_training_data()
    print(f"✓ Created {len(train_data)} training examples")
    
    # Step 3: Initialize model for fine-tuning
    lm = dspy.LM(
        model="openai/gpt-3.5-turbo",
        api_key="your-api-key-here",
        finetuning_model="gpt-3.5-turbo",
        temperature=0.0
    )
    dspy.settings.configure(lm=lm)
    print("✓ Model initialized")
    
    # Step 4: Start fine-tuning
    try:
        print("Starting fine-tuning job...")
        training_job = lm.finetune(
            train_data=train_data,
            train_data_format="openai_chat",
            train_kwargs={
                "n_epochs": 3,
                "batch_size": 1,
                "learning_rate_multiplier": 0.1,
                "suffix": "field-mapping-v1"
            }
        )
        
        print("Fine-tuning in progress...")
        fine_tuned_lm = training_job.result()  # This blocks until completion
        
        print("✓ Fine-tuning completed!")
        print(f"✓ Fine-tuned model: {fine_tuned_lm.model}")
        
        # Step 5: Test the fine-tuned model
        test_questions = [
            "What Trend Micro field maps to Microsoft Defender SHA256?",
            "Map Microsoft Defender ProcessName to Trend Micro schema",
            "What is the Trend Micro equivalent of IPAddress?"
        ]
        
        print("\n=== Testing Fine-tuned Model ===")
        for question in test_questions:
            response = fine_tuned_lm(question)
            print(f"Q: {question}")
            print(f"A: {response[0]}")
            print()
        
        return fine_tuned_lm
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        return None

# Example 5: Comparison between fine-tuning and prompt optimization
def compare_approaches():
    """
    Compare fine-tuning vs prompt optimization approaches.
    """
    
    print("=== Fine-tuning vs Prompt Optimization ===\n")
    
    print("FINE-TUNING (Modifies model weights):")
    print("✓ Changes the model's internal parameters")
    print("✓ Model learns patterns from training data")
    print("✓ Better for domain-specific knowledge")
    print("✓ Requires experimental flag in DSPy")
    print("✓ Takes longer and costs more")
    print("✓ Results in a new model")
    print()
    
    print("PROMPT OPTIMIZATION (Optimizes prompting strategy):")
    print("✓ Finds better ways to prompt the same model")
    print("✓ No model weights are changed")
    print("✓ Faster and cheaper")
    print("✓ Good for improving reasoning patterns")
    print("✓ Uses optimizers like MIPROv2, BootstrapFewShot")
    print("✓ Results in better prompts/examples")
    print()
    
    print("For your field mapping use case:")
    print("- Fine-tuning: Better if you have lots of field mapping data")
    print("- Prompt optimization: Better for few-shot learning with examples")

if __name__ == "__main__":
    print("DSPy Fine-tuning Examples")
    print("=" * 50)
    
    # Uncomment the example you want to run:
    # example_dspy_finetuning()
    # example_prompt_optimization()
    # complete_finetuning_workflow()
    compare_approaches()
