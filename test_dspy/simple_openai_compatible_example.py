import dspy

# Simple example for quick testing
def quick_example():
    """Quick and simple example to get started"""
    
    # Replace these with your actual values
    LLM_API_ENDPOINT = "https://your-api-endpoint.com/v1"
    LLM_API_KEY = "your-api-key-here"
    
    # Initialize the language model
    lm = dspy.LM(
        model="gpt-4o-mini",                # Your model name
        api_base=LLM_API_ENDPOINT,          # Your OpenAI-compatible endpoint
        api_key=LLM_API_KEY,                # Your API key
        temperature=0.7,
        max_tokens=1000
    )
    
    # Configure DSPy to use this language model
    dspy.settings.configure(lm=lm)
    
    # Test the connection with a simple call
    try:
        response = lm("Hello! Can you tell me a fun fact about Python programming?")
        print("✅ Connection successful!")
        print("Response:", response[0])
        
        # Show usage statistics
        if lm.history:
            last_call = lm.history[-1]
            print(f"\nUsage: {last_call['usage']}")
            
    except Exception as e:
        print("❌ Connection failed:")
        print(f"Error: {e}")

if __name__ == "__main__":
    quick_example()
