# DSPy OpenAI-Compatible API Examples

This directory contains examples for using DSPy with OpenAI-compatible APIs.

## Quick Start

1. **Basic Setup**: Use `simple_openai_compatible_example.py` for a quick test
2. **Comprehensive Examples**: Check `openai_compatible_examples.py` for detailed usage patterns

## Key Configuration Parameters

When using DSPy with OpenAI-compatible APIs, you need to configure these main parameters:

- `model`: The model name (e.g., "gpt-4o-mini", "gpt-3.5-turbo")
- `api_base`: Your OpenAI-compatible endpoint URL
- `api_key`: Your API authentication key
- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = very random)
- `max_tokens`: Maximum tokens to generate per response

## Basic Usage Pattern

```python
import dspy

# Initialize the language model
lm = dspy.LM(
    model="gpt-4o-mini",
    api_base="https://your-api-endpoint.com/v1",
    api_key="your-api-key-here",
    temperature=0.7,
    max_tokens=1000
)

# Configure DSPy
dspy.settings.configure(lm=lm)

# Make a call
response = lm("Your prompt here")
print(response[0])
```

## Environment Variables

You can also use environment variables for configuration:

```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_API_BASE="https://your-api-endpoint.com/v1"
```

Then in your code:
```python
lm = dspy.LM(model="gpt-4o-mini")  # Will pick up env vars automatically
```

## Advanced Features

- **Caching**: Enable with `cache=True` for better performance
- **Retries**: Configure with `num_retries=N` for reliability
- **Multiple Models**: Use different models for different tasks
- **History Tracking**: Access call history via `lm.history`
- **DSPy Modules**: Use with DSPy signatures and predictors

## Files in This Directory

- `simple_openai_compatible_example.py`: Quick start example
- `openai_compatible_examples.py`: Comprehensive examples covering all features
- `README_openai_compatible.md`: This documentation file

## Common Use Cases

1. **Simple Text Generation**: Direct prompts for basic tasks
2. **Chat-Style Interactions**: Using message format with system/user roles
3. **DSPy Modules**: Integration with DSPy signatures and predictors
4. **Multiple Models**: Using different models for different complexity levels
5. **Error Handling**: Robust error handling with retries

## Tips

- Start with the simple example to test your connection
- Use caching to improve performance and reduce costs
- Configure appropriate retry settings for production use
- Monitor usage statistics via the history feature
- Use environment variables for secure credential management
