from llama_cpp import Llama

# Load the GGUF model from the models subfolder
llm = Llama(
    model_path="./models/gemma-2-2b-it-Q8_0.gguf",
    n_ctx=2048,      # adjust context window if needed
    n_threads=8,     # adjust based on your CPU cores
    verbose=True
)

# Test prompt
prompt = "Explain what Yosemite National Park is known for."

# Run inference
output = llm(prompt)

# Print response
print("\n=== Model Output ===")
print(output["choices"][0]["text"].strip())
