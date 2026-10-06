import os
import re
from typing import List
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
import config

def pre_process_code_string(text: str) -> str:
    """
    Code-aware pre-processing:
    Splits camelCase, PascalCase, snake_case, and isolates symbols/punctuation.
    """
    if not text:
        return ""
    # Split camelCase & PascalCase: e.g. "calculateArea" -> "calculate Area"
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    # Split snake_case: e.g. "calculate_area" -> "calculate area"
    text = text.replace("_", " ")
    # Insert spaces around punctuation symbols
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class ScratchCodeTokenizer:
    """
    Custom 100% scratch BPE Subword Tokenizer trained strictly on code and text datasets.
    """
    def __init__(self, vocab_size: int = config.VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self._build_blank_tokenizer()

    def _build_blank_tokenizer(self):
        # Initialize BPE tokenizer from scratch
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        # Pre-tokenization: whitespace and byte-level splitting
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation()
        ])
        tokenizer.decoder = decoders.BPEDecoder()
        self.tokenizer = tokenizer

    def train_from_texts(self, texts: List[str], save_path: str = None):
        """
        Train BPE vocabulary from scratch on provided text corpus.
        """
        processed_texts = [pre_process_code_string(t) for t in texts if t.strip()]
        if not processed_texts:
            processed_texts = ["def main(): pass", "function add(a, b) { return a + b; }"]

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            min_frequency=2
        )

        self.tokenizer.train_from_iterator(processed_texts, trainer=trainer)

        # Post-processor to add CLS and SEP tokens
        cls_token_id = self.tokenizer.token_to_id("[CLS]")
        sep_token_id = self.tokenizer.token_to_id("[SEP]")

        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", cls_token_id),
                ("[SEP]", sep_token_id),
            ],
        )

        if save_path:
            self.save(save_path)

    def encode(self, text: str, max_length: int = config.MAX_SEQ_LEN) -> List[int]:
        """
        Tokenize string and return fixed-length token IDs array.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded!")

        processed = pre_process_code_string(text)
        encoding = self.tokenizer.encode(processed)
        ids = encoding.ids

        pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = 0

        # Truncate or pad
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [pad_id] * (max_length - len(ids))

        return ids

    def encode_batch(self, texts: List[str], max_length: int = config.MAX_SEQ_LEN) -> List[List[int]]:
        return [self.encode(t, max_length=max_length) for t in texts]

    def save(self, path: str = config.TOKENIZER_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)
        print(f"Custom Scratch Tokenizer saved to: {path}")

    def load(self, path: str = config.TOKENIZER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found at: {path}")
        self.tokenizer = Tokenizer.from_file(path)
        print(f"Custom Scratch Tokenizer loaded from: {path}")
