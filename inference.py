import json
import torch
import jiwer
from tqdm import tqdm
from transformers import BartForConditionalGeneration
from google.colab import files  # Added for automatic download

# Local imports
from tokenization import BPETokenizer, UnigramTokenizer, CharTokenizer
from train_blstm_att_bpe import Seq2SeqBLSTMAttention

# ==========================================
# 1. CENTRAL CONFIGURATION (REGISTRY)
# ==========================================
MODEL_REGISTRY = {
    "blstm_att_bpe": {
        "arch": "blstm",
        "tok": "bpe",
        "path": "best_models/blstm_bpe_best_model.pth"
    },
    "blstm_att_unigram": {
        "arch": "blstm",
        "tok": "unigram",
        "path": "best_models/blstm_att_unigram_best.pth"
    },
    "bart_bpe": {
        "arch": "bart",
        "tok": "bpe",
        "path": "best_models/bart_bpe_best_model"
    },
    "bart_unigram": {
        "arch": "bart",
        "tok": "unigram",
        "path": "best_models/bart_unigram_best_model"
    }
}

def run_inference(selected_model: str):
    """
    Runs inference by dynamically loading the specified model.
    """
    if selected_model not in MODEL_REGISTRY:
        print(f"Error: The model '{selected_model}' does not exist.")
        print(f"Choose from: {list(MODEL_REGISTRY.keys())}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODEL_REGISTRY[selected_model]
    
    print(f"Starting Inference | Model: {selected_model.upper()} | Device: {device}")

    # ==========================================
    # 2. OBJECT INITIALIZATION
    # ==========================================
    if config["tok"] == "bpe":
        encoder_tokenizer = BPETokenizer("bpe_english.model")
    else:
        encoder_tokenizer = UnigramTokenizer("unigram_english.model")
        
    decoder_tokenizer = CharTokenizer("char_italian.model")

    if config["arch"] == "bart":
        model = BartForConditionalGeneration.from_pretrained(config["path"])
    else:
        enc_vocab = encoder_tokenizer.sp.get_piece_size()
        dec_vocab = decoder_tokenizer.sp.get_piece_size()
        model = Seq2SeqBLSTMAttention(
            enc_vocab_size=enc_vocab, 
            dec_vocab_size=dec_vocab, 
            pad_idx=encoder_tokenizer.sp.pad_id()
        )
        model.load_state_dict(torch.load(config["path"], map_location=device))

    model.to(device)
    model.eval()

    # ==========================================
    # 3. FILE READING AND GENERATION
    # ==========================================
    TEST_FILE = "test.jsonl" 
    
    predictions = []
    references = []
    
    # Variables to save logs and count prints
    output_logs = []
    printed_count = 0
    MAX_PRINTS = 15

    print("\nGenerating transliterations...")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines() 
                
        for line in tqdm(lines, desc="Processing", leave=False):
            if not line.strip():
                continue
                
            data = json.loads(line.strip())
            english_word = data['english_word']
            ground_truth = data['italian_transliteration']
            
            input_ids = torch.tensor([encoder_tokenizer.encode(english_word)]).to(device)
            attention_mask = (input_ids != encoder_tokenizer.sp.pad_id()).long().to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=50,
                    bos_token_id=decoder_tokenizer.sp.bos_id(),
                    eos_token_id=decoder_tokenizer.sp.eos_id(),
                    pad_token_id=decoder_tokenizer.sp.pad_id()
                )
            
            gen_ids = generated_ids[0]
            clean_ids = []
            for t_id in gen_ids:
                t_id = t_id.item()
                if t_id == decoder_tokenizer.sp.eos_id():
                    break
                if t_id not in [decoder_tokenizer.sp.pad_id(), decoder_tokenizer.sp.bos_id()]:
                    clean_ids.append(t_id)
                    
            prediction = decoder_tokenizer.decode(clean_ids)
            
            # Create the formatted string
            log_line = f"EN: {english_word:<15} | TRUE: {ground_truth:<15} | PRED: {prediction}"
            output_logs.append(log_line)
            
            # Print only the first 15 words to the screen
            if printed_count < MAX_PRINTS:
                print(log_line)
                printed_count += 1
            
            predictions.append(" ".join(list(prediction)))
            references.append(" ".join(list(ground_truth)))

    # ==========================================
    # 4. EVALUATION AND FILE SAVING
    # ==========================================
    final_cer = jiwer.wer(references, predictions)
    
    header_result = f"\n--- FINAL RESULT ({selected_model.upper()}) ---"
    text_cer = f"Character Error Rate (CER): {final_cer:.4f}"
    
    print(header_result)
    print(text_cer)
    
    # TXT file generation
    output_filename = f"inference_{selected_model}.txt"
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        out_f.write(f"Inference Report - Model: {selected_model.upper()}\n")
        out_f.write("=" * 60 + "\n")
        for log in output_logs:
            out_f.write(log + "\n")
        out_f.write("=" * 60 + "\n")
        out_f.write(header_result.strip() + "\n")
        out_f.write(text_cer + "\n")
        
    print(f"\nDownloading file '{output_filename}'...")
    files.download(output_filename)