import json
import torch
import torch.nn as nn
import jiwer
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm import tqdm
from tokenization import BPETokenizer, CharTokenizer

#1. Dataset class to read the JSONL file and provide pairs of (english_word, italian_transliteration)
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

#2. Collate function to handle variable-length sequences and padding
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

#3. Model configuration: Bidirectional LSTM with Attention 
class Seq2SeqBLSTMAttention(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, pad_idx, hidden_size=256, num_layers=1):
        super().__init__()
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size
        
        # Encoder (Bidirectional)
        self.enc_embedding = nn.Embedding(enc_vocab_size, hidden_size, padding_idx=pad_idx)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        
        # Attention Layers (Bahdanau)
        # Receives hidden state of the decoder (hidden_size) + encoder output (hidden_size * 2)
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
        # Decoder (Unidirectional)
        self.dec_embedding = nn.Embedding(dec_vocab_size, hidden_size, padding_idx=pad_idx)
        # Decoder input = current embedding (hidden_size) + context vector from attention (hidden_size * 2)
        self.decoder = nn.LSTM(hidden_size * 3, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, dec_vocab_size)

        # Layers to merge the bidirectional hidden states of the encoder and pass them to the decoder
        self.hidden_transform = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_transform = nn.Linear(hidden_size * 2, hidden_size)
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def _calculate_attention(self, hidden, enc_outputs, attention_mask):
        # hidden_state: (1, batch_size, hidden_size) -> (batch_size, 1, hidden_size)
        h_t = hidden[-1].unsqueeze(1).repeat(1, enc_outputs.size(1), 1)
        
        # Compute energy scores and attention weights
        energy = torch.tanh(self.attn(torch.cat((h_t, enc_outputs), dim=2)))
        attention_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)
        
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask == 0, 1e-10)
            attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True)
            
        # Context vector
        context = torch.bmm(attention_weights.unsqueeze(1), enc_outputs)
        return context

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.size(0)
        enc_embeds = self.enc_embedding(input_ids)
        
        # Pass through the encoder
        enc_outputs, (hidden, cell) = self.encoder(enc_embeds)
        
        # Unify the forward and backward directions of the encoder to initialize the decoder
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell_cat = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        
        hidden = torch.tanh(self.hidden_transform(hidden_cat)).unsqueeze(0)
        cell = torch.tanh(self.cell_transform(cell_cat)).unsqueeze(0)
        
        dec_input = labels[:, :-1].clone()
        dec_input[dec_input == -100] = self.pad_idx 
        seq_len = dec_input.size(1)
        
        dec_embeds = self.dec_embedding(dec_input)
        logits = torch.zeros(batch_size, seq_len, self.fc_out.out_features).to(input_ids.device)
        
        # Loop through each time step for Teacher Forcing with Attention
        for t in range(seq_len):
            context = self._calculate_attention(hidden, enc_outputs, attention_mask)
            rnn_input = torch.cat((dec_embeds[:, t:t+1, :], context), dim=2)
            out, (hidden, cell) = self.decoder(rnn_input, (hidden, cell))
            logits[:, t, :] = self.fc_out(out.squeeze(1))
            
        loss = None
        if labels is not None:
            target = labels[:, 1:].contiguous().view(-1)
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), target)
            
        class Output: pass
        out = Output()
        out.loss = loss
        out.logits = logits
        return out

    def generate(self, input_ids, attention_mask, max_length, bos_token_id, eos_token_id, pad_token_id):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        enc_embeds = self.enc_embedding(input_ids)
        enc_outputs, (hidden, cell) = self.encoder(enc_embeds)
        
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell_cat = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        
        hidden = torch.tanh(self.hidden_transform(hidden_cat)).unsqueeze(0)
        cell = torch.tanh(self.cell_transform(cell_cat)).unsqueeze(0)
        
        dec_input = torch.tensor([[bos_token_id]] * batch_size, device=device)
        generated_ids = []
        
        for _ in range(max_length):
            dec_embeds = self.dec_embedding(dec_input)
            context = self._calculate_attention(hidden, enc_outputs, attention_mask)
            
            rnn_input = torch.cat((dec_embeds, context), dim=2)
            output, (hidden, cell) = self.decoder(rnn_input, (hidden, cell))
            
            logits = self.fc_out(output[:, -1, :])
            next_token = logits.argmax(1).unsqueeze(1)
            generated_ids.append(next_token)
            dec_input = next_token
            
        return torch.cat(generated_ids, dim=1)

#4. Evaluation function to compute loss and CER on the validation set
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

            # Decode generation output and truncate at <eos>
            for gen_ids, ref_text in zip(generated_ids, target_texts):
                clean_ids = []
                for token_id in gen_ids:
                    t_id = token_id.item()
                    # if we reach the <eos> token, we stop decoding
                    if t_id == decoder_tokenizer.sp.eos_id():
                        break
                    # if we encounter padding or bos tokens, we ignore them
                    if t_id not in [decoder_tokenizer.sp.pad_id(), decoder_tokenizer.sp.bos_id()]:
                        clean_ids.append(t_id)
                
                pred_text = decoder_tokenizer.decode(clean_ids)
                all_predictions.append(" ".join(list(pred_text)))
                all_references.append(" ".join(list(ref_text)))

    avg_loss = total_loss / len(dataloader)
    cer = jiwer.wer(all_references, all_predictions) if all_references else 0.0
    return avg_loss, cer

#5. Training function to orchestrate the training process
def train_blstm_architecture():
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

    enc_vocab = encoder_tokenizer.sp.get_piece_size()
    dec_vocab = decoder_tokenizer.sp.get_piece_size()
    
    model = Seq2SeqBLSTMAttention(
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

        # Save model weights after each epoch starting from the 4th epoch
        if epoch >= 4: 
            epoch_filename = f"blstm_bpe_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), epoch_filename)
            print(f"Progressive saving: weights from epoch {epoch+1} saved to '{epoch_filename}'")
        # ==========================================

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            patience_counter = 0
            torch.save(model.state_dict(), "blstm_bpe_best_model.pth")
            print(f"New best model found (CER: {best_val_cer:.4f})! Weights serialized.")
        else:
            patience_counter += 1
            print(f"No improvement in CER. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

if __name__ == "__main__":
    train_blstm_architecture()