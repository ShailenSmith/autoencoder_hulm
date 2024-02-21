from transformers import AutoModelForCausalLM

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
print("loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    # torch_dtype=bfloat16,
    # device_map='auto'
)
print("model loaded")