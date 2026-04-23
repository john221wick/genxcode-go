from transformers import AutoTokenizer


class Tokenizer:
    """Wrapper around Hugging Face tokenizer"""

    def __init__(self, model_name: str = "gpt2"):
        self.tok = AutoTokenizer.from_pretrained(model_name)

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    @property
    def eos_token_id(self) -> int:
        return self.tok.eos_token_id
