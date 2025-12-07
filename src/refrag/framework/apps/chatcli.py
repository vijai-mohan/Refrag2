from dataclasses import dataclass
from omegaconf import DictConfig

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as torch


@dataclass
class ChatCLIApp:

    def __init__(self, **kwargs):
        self.model_name = kwargs["model_name"]
        self.device = kwargs["device"]
        self.system_prompt = kwargs.get("system_prompt", None)

        print(f"Loading model: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.conversation_history = []

        # Add system prompt to conversation history if provided
        if self.system_prompt:
            self.conversation_history.append({'role': 'system', 'content': self.system_prompt})
            print(f"System prompt: {self.system_prompt}")

        print(f"Ready! Type your message or 'quit' to exit.\n")

    def _apply_chat_template(self) -> str:
        """Apply chat template to conversation history."""
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                return self.tokenizer.apply_chat_template(
                    self.conversation_history,
                    tokenize=False,
                    add_generation_prompt=True
                )
        except:
            pass

        # Simple fallback
        parts = []
        for msg in self.conversation_history:
            parts.append(f"{msg['role']}: {msg['content']}")
        parts.append("assistant:")
        return "\n".join(parts)

    def run(self, cfg: DictConfig):
        """Run the interactive chat loop."""
        while True:
            try:
                prompt = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            prompt = prompt.strip()
            if not prompt:
                continue

            if prompt.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            elif prompt == "/clear":
                self.conversation_history = []
                # Re-add system prompt if it exists
                if self.system_prompt:
                    self.conversation_history.append({'role': 'system', 'content': self.system_prompt})
                print("Conversation cleared.")
                continue
            elif prompt == "/history":
                if not self.conversation_history:
                    print("No history yet.")
                else:
                    for i, msg in enumerate(self.conversation_history):
                        print(f"{i+1}. {msg['role']}: {msg['content']}")
                continue

            # Add user message
            self.conversation_history.append({'role': 'user', 'content': prompt})

            try:
                # Generate response
                full_prompt = self._apply_chat_template()
                inputs = self.tokenizer(full_prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    )

                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract response
                if full_text.startswith(full_prompt):
                    response = full_text[len(full_prompt):].strip()
                else:
                    response = full_text.strip()

                print(f"Assistant: {response}\n")

                # Add to history
                self.conversation_history.append({'role': 'assistant', 'content': response})

            except Exception as e:
                print(f"Error: {e}\n")
                self.conversation_history.pop()  # Remove failed user message
