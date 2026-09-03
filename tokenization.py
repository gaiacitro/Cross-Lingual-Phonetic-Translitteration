import sentencepiece as spm
import os

#1. BPETokenizer class for English input using Byte Pair Encoding (BPE)
class BPETokenizer:
    def __init__(self, model_path: str = "bpe_english.model"):
        self.sp = spm.SentencePieceProcessor()
        if os.path.exists(model_path):
            self.sp.load(model_path)
            
    @classmethod
    def train(cls, input_file: str, model_prefix: str = "bpe_english", vocab_size: int = 300):
        print(f"Training BPE Tokenizer with vocab_size={vocab_size}...")
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>"
        )
        print(f"BPE model saved as {model_prefix}.model")

    def encode(self, text: str) -> list[int]:
        # Encode text to IDs and attach beginning-of-sequence and end-of-sequence tokens
        return [self.sp.bos_id()] + self.sp.encode_as_ids(text) + [self.sp.eos_id()]
        
    def decode(self, ids: list[int]) -> str:
        return self.sp.decode_ids(ids)

#2. UnigramTokenizer class for English input using probabilistic subword modeling
class UnigramTokenizer:
    def __init__(self, model_path: str = "unigram_english.model"):
        self.sp = spm.SentencePieceProcessor()
        if os.path.exists(model_path):
            self.sp.load(model_path)
            
    @classmethod
    def train(cls, input_file: str, model_prefix: str = "unigram_english", vocab_size: int = 300):
        print(f"Training Unigram Tokenizer with vocab_size={vocab_size}...")
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="unigram",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>"
        )
        print(f"Unigram model saved as {model_prefix}.model")

    def encode(self, text: str, enable_sampling: bool = False, alpha: float = 0.1) -> list[int]:
        # Encode with optional subword regularization sampling for training
        ids = self.sp.encode(text, enable_sampling=enable_sampling, alpha=alpha, out_type=int)
        return [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        
    def decode(self, ids: list[int]) -> str:
        return self.sp.decode_ids(ids)

#3. CharTokenizer class for Italian TFI target output at character level
class CharTokenizer:
    def __init__(self, model_path: str = "char_italian.model"):
        self.sp = spm.SentencePieceProcessor()
        if os.path.exists(model_path):
            self.sp.load(model_path)
            
    @classmethod
    def train(cls, input_file: str, model_prefix: str = "char_italian", vocab_size: int = 40):
        print("Training Character-level Tokenizer for Italian TFI...")
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="char", # Forces character-level split
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>"
        )
        print(f"Character model saved as {model_prefix}.model")

    def encode(self, text: str) -> list[int]:
        # Encode character sequence to IDs with bos and eos tokens
        return [self.sp.bos_id()] + self.sp.encode_as_ids(text) + [self.sp.eos_id()]

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode_ids(ids)

#4. Main execution entry point to train and save all tokenizers
if __name__ == "__main__":
    english_txt = "english_words.txt"
    italian_txt = "italian_transliterations.txt"
    
    if os.path.exists(english_txt) and os.path.exists(italian_txt):
        print("Starting training of all tokenizers...")
        BPETokenizer.train(english_txt)
        UnigramTokenizer.train(english_txt)
        CharTokenizer.train(italian_txt)
        print("All tokenizers have been created successfully")
    else:
        print("Error: Input text files not found.")