import json
import torch
import jiwer
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import BartConfig, BartForConditionalGeneration
from torch.optim import AdamW
from tqdm import tqdm

# Local module imports
from tokenization import BPETokenizer, CharTokenizer

# ==========================================
# 1. DATASET MANAGEMENT
# ==========================================
class TransliterationDataset(Dataset):
    """
    Dataset class for loading English-to-Italian phonetic transliteration pairs.
    Parses a JSONL file containing 'english_word' and 'italian_transliteration'.
    """
    def __init__(self, jsonl_file_path: str):
        self.data_pairs = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line.strip())
                self.data_pairs.append((record['english_word'], record['italian_transliteration']))
                
    def __len__(self) -> int:
        return len(self.data_pairs)
        
    def __getitem__(self, index: int) -> tuple:
        return self.data_pairs[index]

# ==========================================
# 2. COLLATION UTILITIES
# ==========================================
def get_collate_fn(encoder_tokenizer: BPETokenizer, decoder_tokenizer: CharTokenizer):
    """
    Returns a collation function that dynamically tokenizes text and pads sequences.
    Replaces target padding tokens with -100 to mask them during CrossEntropyLoss computation.
    """
    def collate_fn(batch: list) -> dict:
        source_texts = [item[0] for item in batch]
        target_texts = [item[1] for item in batch]

        # Dynamic tokenization (encoding sequence generation)
        source_ids = [torch.tensor(encoder_tokenizer.encode(text)) for text in source_texts]
        target_ids = [torch.tensor(decoder_tokenizer.encode(text)) for text in target_texts]

        pad_id_source = encoder_tokenizer.sp.pad_id()
        pad_id_target = decoder_tokenizer.sp.pad_id()

        # Sequence padding for uniform batch dimensionality
        input_ids = pad_sequence(source_ids, batch_first=True, padding_value=pad_id_source)
        labels = pad_sequence(target_ids, batch_first=True, padding_value=pad_id_target)

        # Attention mask creation (1 for valid tokens, 0 for padding tokens)
        attention_mask = (input_ids != pad_id_source).long()

        # Mask padding in labels to omit them from the loss gradient computation
        labels[labels == pad_id_target] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'target_texts': target_texts # Preserved as raw text for CER evaluation
        }
    return collate_fn

# ==========================================
# 3. EVALUATION METRICS
# ==========================================
def evaluate_model(model, dataloader, device, decoder_tokenizer) -> tuple:
    """
    Evaluates the model on the validation set, computing average validation loss 
    and Character Error Rate (CER).
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            target_texts = batch['target_texts']

            # Forward pass for loss computation
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            # Autoregressive generation for metric computation
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                bos_token_id=decoder_tokenizer.sp.bos_id(),
                eos_token_id=decoder_tokenizer.sp.eos_id(),
                pad_token_id=decoder_tokenizer.sp.pad_id()
            )

            # Decode generation output and compute Levenshtein distance components
            for gen_ids, ref_text in zip(generated_ids, target_texts):
                clean_ids = [
                    token_id.item() for token_id in gen_ids 
                    if token_id not in [decoder_tokenizer.sp.pad_id(), 
                                        decoder_tokenizer.sp.bos_id(), 
                                        decoder_tokenizer.sp.eos_id()]
                ]
                pred_text = decoder_tokenizer.decode(clean_ids)
                
                # To calculate CER using jiwer's WER function, we inject spaces between characters
                all_predictions.append(" ".join(list(pred_text)))
                all_references.append(" ".join(list(ref_text)))

    avg_loss = total_loss / len(dataloader)
    
    # Compute the final CER
    cer = jiwer.wer(all_references, all_predictions) if all_references else 0.0
    
    return avg_loss, cer

# ==========================================
# 4. TRAINING PIPELINE
# ==========================================
def train_bart_architecture():
    """
    Main execution pipeline for training the custom BART architecture.
    Implements an 80/20 train-validation data split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on computational device: {device}")

    # Initialize Tokenization Objects
    encoder_tokenizer = BPETokenizer("bpe_english.model")
    decoder_tokenizer = CharTokenizer("char_italian.model")

    # Dataset Initialization and Partitioning
    full_dataset = TransliterationDataset("transliteration_dataset.jsonl")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Fix the generator seed to guarantee partition reproducibility across iterations
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    collate_function = get_collate_fn(encoder_tokenizer, decoder_tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_function)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_function)

    print(f"Dataset partitioned. Training instances: {train_size} | Validation instances: {val_size}")

    # Ensure the shared vocabulary matrix can hold the largest tokenizer's IDs
    max_vocab_size = max(encoder_tokenizer.sp.get_piece_size(), decoder_tokenizer.sp.get_piece_size())

    # Model Configuration
    configuration = BartConfig(
        vocab_size=max_vocab_size,
        d_model=256,
        encoder_layers=4,
        decoder_layers=4,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        max_position_embeddings=50,
        pad_token_id=encoder_tokenizer.sp.pad_id(),
        bos_token_id=encoder_tokenizer.sp.bos_id(),
        eos_token_id=encoder_tokenizer.sp.eos_id(),
        forced_eos_token_id=decoder_tokenizer.sp.eos_id()
    )
    
    model = BartForConditionalGeneration(configuration)
    model.to(device)

    # Optimization Setup
    optimizer = AdamW(model.parameters(), lr=5e-4)
    num_epochs = 10

    # Early Stopping Configuration
    patience = 3
    patience_counter = 0
    best_val_cer = float('inf')  # We monitor CER as it's the target metric

    # Training and Evaluation Epoch Iteration
    for epoch in range(num_epochs):
        # --- Phase 1: Training ---
        model.train()
        total_train_loss = 0.0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch in train_iterator:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_iterator.set_postfix(loss=loss.item())
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Phase 2: Validation ---
        val_loss, val_cer = evaluate_model(model, val_loader, device, decoder_tokenizer)

        print(f"Epoch {epoch+1}/{num_epochs} Completed | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f}")

        # --- Phase 3: Early Stopping & Model Checkpointing ---
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            patience_counter = 0
            # Save only when we achieve a new best validation CER
            model.save_pretrained("./bart_bpe_best_model")
            print(f"New best model found (CER: {best_val_cer:.4f})! Weights serialized.")
        else:
            patience_counter += 1
            print(f"No improvement in CER. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Halting training protocol.")
                break

    print("Training protocol concluded. The optimal model state is preserved in './bart_bpe_best_model'.")

if __name__ == "__main__":
    train_bart_architecture()