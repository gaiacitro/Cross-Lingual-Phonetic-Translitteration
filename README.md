# Cross-Lingual Phonetic Transliteration (English → Italian)

Sequence-to-sequence models that convert English words into **Italian-readable phonetic graphemes** — i.e. a spelling of the English pronunciation that a native Italian speaker can read aloud correctly, without needing to know IPA or ARPABET.

Example: `friendly` → `frendli`, `time` → `taim`, `thought` → `tot`.

This repository contains the full pipeline: dataset generation from the CMU Pronouncing Dictionary, a deterministic ARPABET→Italian grapheme mapping, subword tokenizer training (BPE / Unigram / character-level), and training scripts for two sequence-to-sequence architectures — a **BLSTM with Bahdanau attention** (trained from scratch in PyTorch) and a **compact BART** (trained from scratch via Hugging Face Transformers).

Developed for the NLP course project, Sapienza University of Rome (2024–2025).

---

## Repository structure

| File | Description |
|---|---|
| `mapping_cmu_italian.py` | Deterministic, rule-based mapping from stress-free ARPABET phonemes to Italian graphemes, including context-aware (lookahead) handling of velars (`K`/`G`) and affricates (`CH`/`JH`). |
| `dataset_creation.py` | Parses the raw CMU Pronouncing Dictionary, strips stress markers, applies the mapping above, and writes the parallel dataset. |
| `tokenization.py` | Trains and wraps SentencePiece tokenizers: `BPETokenizer` and `UnigramTokenizer` for the English source side, `CharTokenizer` (character-level) for the Italian target side. |
| `train_blstm_att_bpe.py` | Trains the custom BLSTM+Attention model with BPE-tokenized English input. |
| `train_blstm_att_unigram.py` | Trains the custom BLSTM+Attention model with Unigram-tokenized English input (subword regularization enabled during training). |
| `train_bart_bpe.py` | Trains a from-scratch BART (Hugging Face `BartForConditionalGeneration`) with BPE-tokenized English input. |
| `train_bart_unigram.py` | Trains a from-scratch BART with Unigram-tokenized English input. |

---

## Pipeline

The scripts are meant to be run in this order:

```bash
# 1. Generate the parallel dataset from the CMU Pronouncing Dictionary
#    (requires cmudict.dict in the working directory — see "Data" below)
python dataset_creation.py

# 2. Train the SentencePiece tokenizers (BPE + Unigram for English, char-level for Italian)
python tokenization.py

# 3. Train the models (any subset, independently)
python train_blstm_att_bpe.py
python train_blstm_att_unigram.py
python train_bart_bpe.py
python train_bart_unigram.py
```

### 1. Dataset generation — `dataset_creation.py`
Reads `cmudict.dict`, filters out comments/variant markers (e.g. `word(2)`), strips numeric stress from the ARPABET phonemes, and converts each entry into an Italian transliteration via `mapping_cmu_italian.convert_cmu_to_tfi`. Produces:
- `transliteration_dataset.jsonl` — one JSON object per line: `english_word`, `cmu_with_stress`, `cmu_clean`, `italian_transliteration`.
- `english_words.txt` — plain list of English words, used to train the source-side tokenizers.
- `italian_transliterations.txt` — plain list of Italian transliterations, used to train the target-side tokenizer.

### 2. Phoneme-to-grapheme mapping — `mapping_cmu_italian.py`
`convert_cmu_to_tfi(cmu_phonemes: list[str]) -> str` applies:
- A static phoneme→grapheme table for vowels and unambiguous consonants (e.g. `AY`→`ai`, `SH`→`sci`, `OW`→`ou`).
- Context-dependent (lookahead) rules for `CH`/`JH` (soft `c`/`g` vs. `ci`/`gi`) and `K`/`G` (hard `ch`/`gh` vs. `c`/`g`), based on whether the following phoneme is a front vowel, a back vowel/consonant, or the end of the word.

### 3. Tokenizer training — `tokenization.py`
Trains three SentencePiece models via `spm.SentencePieceTrainer.train`:
- `bpe_english.model` — BPE, `vocab_size=300`, trained on `english_words.txt`.
- `unigram_english.model` — Unigram LM, `vocab_size=300`, trained on `english_words.txt`.
- `char_italian.model` — character-level, `vocab_size=40`, trained on `italian_transliterations.txt`.

All three share the same special-token convention (`pad_id=0, unk_id=1, bos_id=2, eos_id=3`).

### 4. Model training — `train_*.py`
Each script:
- Loads `transliteration_dataset.jsonl` and splits it **80/20** into train/validation with a fixed seed (`torch.Generator().manual_seed(42)`).
- Trains with `AdamW`, learning rate `5e-4`, batch size `64`, for up to `10` epochs, with early stopping (`patience=3`) monitored on **validation CER** (computed with `jiwer`).
- Saves the best checkpoint by validation CER:
  - BLSTM scripts → `blstm_att_bpe_best.pth` / `blstm_att_unigram_best.pth` (raw `state_dict`, via `torch.save`).
  - BART scripts → `./bart_bpe_best_model/` / `./bart_unigram_best_model/` (Hugging Face format, via `model.save_pretrained`).

**BLSTM + Bahdanau Attention** (`train_blstm_att_*.py`): custom PyTorch implementation — single-layer bidirectional LSTM encoder (`hidden_size=256`), Bahdanau-style additive attention, unidirectional LSTM decoder with teacher forcing.

**BART (from scratch)** (`train_bart_*.py`): `BartConfig` with `d_model=256`, `encoder_layers=4`, `decoder_layers=4`, `encoder/decoder_attention_heads=8`, `max_position_embeddings=50` — no pretrained weights loaded.

---

## Data

This repo does **not** include the CMU Pronouncing Dictionary. Before running `dataset_creation.py`, download `cmudict.dict` (version 0.7b) from the [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) and place it in the project root.

> To avoid data leakage between training and the manually curated test set, test-set words and their morphologically related neighbors should be excluded from `cmudict.dict` beforehand (e.g. via comment tags), as described in the accompanying paper.

---

## Requirements

```bash
pip install torch transformers sentencepiece jiwer tqdm
```

A CUDA-capable GPU is recommended (scripts auto-detect `cuda`/`cpu`).

---

## Results

Evaluated on a manually curated hold-out test set of 300 English words, using Character Error Rate (CER):

| Architecture | Tokenizer | Best Val CER | Test CER |
|---|---|---|---|
| BLSTM + Attn | Unigram | 0.0649 | **0.1113** |
| BLSTM + Attn | BPE | 0.0649 | 0.1224 |
| BART (Scratch) | Unigram | 0.0777 | 0.1262 |
| BART (Scratch) | BPE | 0.0772 | 0.1329 |

Full methodology, error analysis, and discussion are available in the accompanying paper.

---

## Authors

- **Gaia Citro** — `citro.2026094@studenti.uniroma1.it`
- **Lucia Fornetti** — `fornetti.2214370@studenti.uniroma1.it`

Sapienza University of Rome — NLP 2024–2025
