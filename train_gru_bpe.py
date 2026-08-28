import json
import torch
import torch.nn as nn
import jiwer
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm import tqdm

from tokenization import BPETokenizer, CharTokenizer

# ==========================================
# 1 & 2. DATASET E COLLATION (IDENTICI A BART)
# ==========================================
class TransliterationDataset(Dataset):
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

def get_collate_fn(encoder_tokenizer, decoder_tokenizer):
    def collate_fn(batch: list) -> dict:
        source_texts = [item[0] for item in batch]
        target_texts = [item[1] for item in batch]

        source_ids = [torch.tensor(encoder_tokenizer.encode(text)) for text in source_texts]
        target_ids = [torch.tensor(decoder_tokenizer.encode(text)) for text in target_texts]

        pad_id_source = encoder_tokenizer.sp.pad_id()
        pad_id_target = decoder_tokenizer.sp.pad_id()

        input_ids = pad_sequence(source_ids, batch_first=True, padding_value=pad_id_source)
        labels = pad_sequence(target_ids, batch_first=True, padding_value=pad_id_target)

        attention_mask = (input_ids != pad_id_source).long()
        labels[labels == pad_id_target] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'target_texts': target_texts
        }
    return collate_fn

# ==========================================
# ARCHITETTURA GRU CUSTOM
# ==========================================
class Seq2SeqGRU(nn.Module):
    """Architettura Encoder-Decoder basata su GRU che simula l'interfaccia HuggingFace."""
    def __init__(self, enc_vocab_size, dec_vocab_size, pad_idx, hidden_size=256, num_layers=2):
        super().__init__()
        self.pad_idx = pad_idx
        
        # Encoder
        self.enc_embedding = nn.Embedding(enc_vocab_size, hidden_size, padding_idx=pad_idx)
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Decoder
        self.dec_embedding = nn.Embedding(dec_vocab_size, hidden_size, padding_idx=pad_idx)
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, dec_vocab_size)
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Passaggio nell'Encoder
        enc_embeds = self.enc_embedding(input_ids)
        _, hidden = self.encoder(enc_embeds)
        
        # Passaggio nel Decoder (Teacher Forcing)
        # Shiftiamo le labels a destra per l'input del decoder
        dec_input = labels[:, :-1].clone()
        dec_input[dec_input == -100] = self.pad_idx # Rimuoviamo i -100 per l'embedding
        
        dec_embeds = self.dec_embedding(dec_input)
        dec_outputs, _ = self.decoder(dec_embeds, hidden)
        logits = self.fc_out(dec_outputs)
        
        loss = None
        if labels is not None:
            # Calcolo della loss confrontando i logits con le labels shiftate
            target = labels[:, 1:].contiguous().view(-1)
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), target)
            
        class Output: pass
        out = Output()
        out.loss = loss
        out.logits = logits
        return out

    def generate(self, input_ids, attention_mask, max_length, bos_token_id, eos_token_id, pad_token_id):
        """Generazione autoregressiva per l'inferenza e la validazione."""
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        enc_embeds = self.enc_embedding(input_ids)
        _, hidden = self.encoder(enc_embeds)
        
        dec_input = torch.tensor([[bos_token_id]] * batch_size, device=device)
        generated_ids = []
        
        for _ in range(max_length):
            dec_embeds = self.dec_embedding(dec_input)
            output, hidden = self.decoder(dec_embeds, hidden)
            logits = self.fc_out(output[:, -1, :])
            next_token = logits.argmax(1).unsqueeze(1)
            
            generated_ids.append(next_token)
            dec_input = next_token
            
        return torch.cat(generated_ids, dim=1)

# ==========================================
# 3. METRICHE E VALUTAZIONE
# ==========================================
def evaluate_model(model, dataloader, device, decoder_tokenizer) -> tuple:
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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                bos_token_id=decoder_tokenizer.sp.bos_id(),
                eos_token_id=decoder_tokenizer.sp.eos_id(),
                pad_token_id=decoder_tokenizer.sp.pad_id()
            )

            for gen_ids, ref_text in zip(generated_ids, target_texts):
                clean_ids = [
                    token_id.item() for token_id in gen_ids 
                    if token_id not in [decoder_tokenizer.sp.pad_id(), 
                                        decoder_tokenizer.sp.bos_id(), 
                                        decoder_tokenizer.sp.eos_id()]
                ]
                pred_text = decoder_tokenizer.decode(clean_ids)
                all_predictions.append(" ".join(list(pred_text)))
                all_references.append(" ".join(list(ref_text)))

    avg_loss = total_loss / len(dataloader)
    cer = jiwer.wer(all_references, all_predictions) if all_references else 0.0
    return avg_loss, cer

# ==========================================
# 4. TRAINING PIPELINE (IDENTICA A BART)
# ==========================================
def train_gru_architecture():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on computational device: {device}")

    encoder_tokenizer = BPETokenizer("bpe_english.model")
    decoder_tokenizer = CharTokenizer("char_italian.model")

    full_dataset = TransliterationDataset("transliteration_dataset.jsonl")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    collate_function = get_collate_fn(encoder_tokenizer, decoder_tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_function)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_function)

    print(f"Dataset partitioned. Training instances: {train_size} | Validation instances: {val_size}")

    # Inizializzazione della nostra architettura GRU
    enc_vocab = encoder_tokenizer.sp.get_piece_size()
    dec_vocab = decoder_tokenizer.sp.get_piece_size()
    
    model = Seq2SeqGRU(
        enc_vocab_size=enc_vocab, 
        dec_vocab_size=dec_vocab, 
        pad_idx=encoder_tokenizer.sp.pad_id()
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-4)
    num_epochs = 10

    patience = 3
    patience_counter = 0
    best_val_cer = float('inf')

    for epoch in range(num_epochs):
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
        val_loss, val_cer = evaluate_model(model, val_loader, device, decoder_tokenizer)

        print(f"Epoch {epoch+1}/{num_epochs} Completed | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f}")

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            patience_counter = 0
            # Salvataggio custom dei pesi di PyTorch
            torch.save(model.state_dict(), "gru_bpe_best_model.pth")
            print(f"New best model found (CER: {best_val_cer:.4f})! Weights serialized.")
        else:
            patience_counter += 1
            print(f"No improvement in CER. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    print("Training protocol concluded. State dictionary preserved as 'gru_bpe_best_model.pth'.")

if __name__ == "__main__":
    train_gru_architecture()