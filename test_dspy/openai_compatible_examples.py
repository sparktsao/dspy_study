import dspy
import os
from typing import List, Dict, Any

# Example 1: Basic OpenAI-compatible API setup
def example_basic_setup():
    """Basic example of setting up DSPy with an OpenAI-compatible API"""
    
    # Initialize the language model with OpenAI-compatible settings
    lm = dspy.LM(
        model="gpt-4o-mini",  # Your model name
        api_base="https://your-api-endpoint.com/v1",  # Your OpenAI-compatible endpoint
        api_key="your-api-key-here",  # Your API key
        temperature=0.7,
        max_tokens=1000,
        cache=True,  # Enable caching for better performance
        num_retries=3  # Number of retries on failure
    )
    
    # Configure DSPy to use this language model
    dspy.settings.configure(lm=lm)
    
    # Simple text generation
    response = lm("What is the capital of France?")
    print("Response:", response[0])
    
    return lm

# Example 2: Using environment variables for configuration
def example_with_env_vars():
    """Example using environment variables for API configuration"""
    
    # Set environment variables (you can also put these in a .env file)
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    os.environ["OPENAI_API_BASE"] = "https://your-api-endpoint.com/v1"
    
    lm = dspy.LM(
        model="gpt-4o-mini",
        # api_key and api_base will be automatically picked up from environment
        temperature=0.0,  # Deterministic responses
        max_tokens=500,
        cache=True
    )
    
    dspy.settings.configure(lm=lm)
    return lm

# Example 3: Advanced configuration with custom parameters
def example_advanced_config():
    """Advanced example with custom parameters and callbacks"""
    
    lm = dspy.LM(
        model="your-custom-model",
        api_base="https://your-api-endpoint.com/v1",
        api_key="your-api-key-here",
        model_type="chat",  # or "text" for text completion models
        temperature=0.8,
        max_tokens=2000,
        cache=True,
        cache_in_memory=True,  # Enable in-memory LRU caching
        num_retries=5,
        # Additional parameters passed to the API
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1
    )
    
    dspy.settings.configure(lm=lm)
    return lm

# Example 4: Different ways to call the language model
def example_different_call_methods():
    """Examples of different ways to call the language model"""
    
    lm = dspy.LM(
        model="gpt-4o-mini",
        api_base="https://your-api-endpoint.com/v1",
        api_key="your-api-key-here"
    )
    
    # Method 1: Simple prompt
    response1 = lm("Explain quantum computing in simple terms.")
    print("Simple prompt response:", response1[0])
    
    # Method 2: Using messages format (chat-style)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    response2 = lm(messages=messages)
    print("Chat format response:", response2[0])
    
    # Method 3: With additional parameters
    response3 = lm(
        "Write a short poem about AI",
        temperature=0.9,  # Override default temperature
        max_tokens=200    # Override default max_tokens
    )
    print("Custom parameters response:", response3[0])
    
    # Method 4: Disable caching for specific call
    response4 = lm("Generate a random number", cache=False)
    print("No cache response:", response4[0])

# Example 5: Using with DSPy modules
def example_with_dspy_modules():
    """Example of using the LM with DSPy modules"""
    
    # Setup the language model
    lm = dspy.LM(
        model="gpt-4o-mini",
        api_base="https://your-api-endpoint.com/v1",
        api_key="your-api-key-here",
        temperature=0.0
    )
    dspy.settings.configure(lm=lm)
    
    # Define a simple DSPy signature
    class BasicQA(dspy.Signature):
        """Answer questions with helpful information."""
        question = dspy.InputField()
        answer = dspy.OutputField()
    
    # Create a predictor using the signature
    qa_predictor = dspy.Predict(BasicQA)
    
    # Use the predictor
    result = qa_predictor(question="What is the largest planet in our solar system?")
    print("DSPy module result:", result.answer)

# Example 6: Error handling and retries
def example_with_error_handling():
    """Example with proper error handling"""
    
    try:
        lm = dspy.LM(
            model="gpt-4o-mini",
            api_base="https://your-api-endpoint.com/v1",
            api_key="your-api-key-here",
            num_retries=3,  # Will retry up to 3 times on transient errors
            temperature=0.7
        )
        
        dspy.settings.configure(lm=lm)
        
        # Make a call with error handling
        response = lm("Explain the theory of relativity")
        print("Success:", response[0])
        
    except Exception as e:
        print(f"Error occurred: {e}")
        # Handle the error appropriately

# Example 7: Multiple models setup
def example_multiple_models():
    """Example of setting up multiple models for different tasks"""
    
    # Fast model for simple tasks
    fast_lm = dspy.LM(
        model="gpt-3.5-turbo",
        api_base="https://your-api-endpoint.com/v1",
        api_key="your-api-key-here",
        temperature=0.0,
        max_tokens=500
    )
    
    # Powerful model for complex tasks
    powerful_lm = dspy.LM(
        model="gpt-4o",
        api_base="https://your-api-endpoint.com/v1",
        api_key="your-api-key-here",
        temperature=0.3,
        max_tokens=2000
    )
    
    # Use different models for different tasks
    dspy.settings.configure(lm=fast_lm)  # Default to fast model
    
    simple_response = fast_lm("What is 2+2?")
    print("Simple math:", simple_response[0])
    
    # Switch to powerful model for complex task
    dspy.settings.configure(lm=powerful_lm)
    complex_response = powerful_lm("Explain the implications of quantum entanglement for computing")
    print("Complex explanation:", complex_response[0])

# Example 8: Accessing model history and usage
def example_history_and_usage():
    """Example of accessing model call history and usage statistics"""
    
    lm = dspy.LM(
        model="gpt-4o-mini",
        api_base="https://your-api-endpoint.com/v1",
        api_key="your-api-key-here"
    )
    
    dspy.settings.configure(lm=lm)
    
    # Make some calls
    lm("Hello, how are you?")
    lm("What is the weather like?")
    
    # Access history
    print(f"Number of calls made: {len(lm.history)}")
    
    # Print details of the last call
    if lm.history:
        last_call = lm.history[-1]
        print(f"Last call timestamp: {last_call['timestamp']}")
        print(f"Last call cost: {last_call.get('cost', 'N/A')}")
        print(f"Last call usage: {last_call['usage']}")

if __name__ == "__main__":
    print("=== DSPy OpenAI-Compatible API Examples ===\n")
    
    # Run examples (uncomment the ones you want to test)
    # example_basic_setup()
    # example_with_env_vars()
    # example_advanced_config()
    # example_different_call_methods()
    # example_with_dspy_modules()
    # example_with_error_handling()
    # example_multiple_models()
    # example_history_and_usage()
    
    print("Examples completed!")
