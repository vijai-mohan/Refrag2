import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
if __name__ == '__main__':
    from huggingface_hub import login

    hf_token = os.environ.get("HF_TOKEN", "hf_fake_token_use_env")
    if not hf_token or hf_token.startswith("hf_fake"):
        raise RuntimeError("Set HF_TOKEN env var to a real Hugging Face token before running this sample.")
    login(hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "ibm-granite/granite-4.0-350M"

    # model_path="gpt2"
    model_path="facebook/opt-125m"
    model_path="meta-llama/Llama-3.2-1B-Instruct"
    #model_path="google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    use_bf16 = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    torch_dtype = torch.bfloat16 if use_bf16 else None
    # drop device_map if running on CPU; use device_map="auto" on GPU
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch_dtype
        )

    # Print a short model summary and parameter counts
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable_params / total_params * 100.0) if total_params else 0.0
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({pct:.2f}%)")

    model.eval()
    # change input text as desired
    chats = [
        {"role": "user",
        # "content": "Please list one IBM Research laboratory located in the United States. You should only output its name and location."
         "content": "What is the capital of india"
        },
    ]

    # Build prompt: use chat template if available; otherwise fallback to a simple prompt
    if getattr(tokenizer, "chat_template", None):
        chat = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)
    else:
        user_msg = next((m["content"] for m in chats if m.get("role") == "user"), "")
        chat = f"User: {user_msg}\nAssistant:"

    #chat = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)
    # tokenize the text
    input_tokens = tokenizer(chat, return_tensors="pt").to(device)
    from transformers import TextIteratorStreamer
    from threading import Thread
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
    generation_kwargs = dict(input_tokens, streamer=streamer, max_new_tokens=20)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    for new_text in streamer:
        print(new_text, end='', flush=True)

    # # generate output tokens
    # output = model.generate(**input_tokens,
    #                         max_new_tokens=100)
    # # decode output tokens into text
    # output = tokenizer.batch_decode(output)
    # # print output
    # print(output[0])
