from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import sys
from typing import Tuple, Optional
import argparse

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_standard_logging():
    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[stdout_handler]
    )

class ConversationExchange:
    def __init__(self, messages: Optional[list[dict]] = None):
        self.messages = messages if messages is not None else []

    def add_user_message(self, message_string: str):
        self.messages.append({"role": "user", "content": message_string}) # type: ignore

    def add_assistant_message(self, message_string: str):
        self.messages.append({"role": "assistant", "content": message_string})

    def get_messages(self) -> list[dict]:
        return self.messages
    
    def get_last_message(self) -> dict:
        return self.messages[-1]


class InfraModel:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        self.model_name = model_name
        self.device = device
        if device != "cuda":
            logger.warning(f"Warning: CUDA is not available. Using {device}.")

        # load the tokenizer and the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def answer_prompt(self, conversation_exchange: ConversationExchange, enable_thinking: bool = False, verbose: bool = False) -> list[Tuple[str, str]]:
        messages = conversation_exchange.get_messages()
        if len(messages) == 0:
            raise ValueError("No messages to answer")
        if verbose:
            logger.info("messages:", messages) # type: ignore
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking, # Switches between thinking and non-thinking modes. Default is True.
        )
        if verbose:
            logger.info("text:", text)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return thinking_content, content

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("user_query", type=str, default="What is the capital of France?")
    return parser.parse_args()

if __name__ == "__main__":
    setup_standard_logging()
    args = parse_args()
    model = InfraModel()
    conversation_exchange = ConversationExchange([{"role": "user", "content": args.user_query}])
    print("conversation_exchange:", conversation_exchange.get_messages())
    thinking_content, content = model.answer_prompt(conversation_exchange, enable_thinking=args.enable_thinking, verbose=False)
    print("thinking_content:", thinking_content)
    print("")
    print("")
    print("content:", content)