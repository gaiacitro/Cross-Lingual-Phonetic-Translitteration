# Cross-Lingual Phonetic Transliteration (English → Italian)

Sequence-to-sequence models that convert English words into **Italian-readable phonetic graphemes** — i.e. a spelling of the English pronunciation that a native Italian speaker can read aloud correctly, without needing to know IPA or ARPABET.

Example: `friendly` → `frendli`, `time` → `taim`, `thought` → `tot`.

This repository contains the full pipeline: dataset generation from the CMU Pronouncing Dictionary, a deterministic ARPABET→Italian grapheme mapping, subword tokenizer training (BPE / Unigram / character-level), and training scripts for two sequence-to-sequence architectures — a **BLSTM with Bahdanau attention** (trained from scratch in PyTorch) and a **compact BART** (trained from scratch via Hugging Face Transformers).

Developed for the NLP course project, Sapienza University of Rome (2024–2025).

---

## Repository structure

**Code**

| File | Description |
|---|---|
| `mapping_cmu_italian.py` | Deterministic, rule-based mapping from stress-free ARPABET phonemes to Italian graphemes, including context-aware (lookahead) handling of velars (`K`/`G`) and affricates (`CH`/`JH`). |
| `dataset_creation.py` | Parses the raw CMU Pronouncing Dictionary, strips stress markers, applies the mapping above, and writes the parallel dataset. |
| `tokenization.py` | Trains and wraps SentencePiece tokenizers: `BPETokenizer` and `UnigramTokenizer` for the English source side, `CharTokenizer` (character-level) for the Italian target side. |
| `train_blstm_att_bpe.py` | Trains the custom BLSTM+Attention model with BPE-tokenized English input. |
| `train_blstm_att_unigram.py` | Trains the custom BLSTM+Attention model with Unigram-tokenized English input (subword regularization enabled during training). |
| `train_bart_bpe.py` | Trains a from-scratch BART (Hugging Face `BartForConditionalGeneration`) with BPE-tokenized English input. |
| `train_bart_unigram.py` | Trains a from-scratch BART with Unigram-tokenized English input. |
| `inference.py` | Loads a trained checkpoint and runs transliteration inference on new/held-out words. |
| *(external)* `Cross_Lingual_Phonetic_Translitteration.ipynb` | Google Colab notebook (hosted on Google Drive, not committed to this repo) that orchestrates the entire pipeline end-to-end on a GPU runtime — see [Running on Google Colab](#running-on-google-colab) below. |

**Data**

| File | Description |
|---|---|
| `cmudict.dict` | The CMU Pronouncing Dictionary (v0.7b) with the entries corresponding to the hold-out test set commented out, so that `dataset_creation.py` excludes them from training — this is the file actually used to build the training data (see [Data](#data) below). |
| `cmudict_complete.dict` | The original, unmodified CMU Pronouncing Dictionary (v0.7b), provided for reference/reproducibility. |
| `test.jsonl` | The manually curated hold-out test set (300 English words, with reference Italian transliterations) used for final evaluation. |
| `english_words.txt` *(generated)* | Plain list of English words, output of `dataset_creation.py`, used to train the source-side tokenizers. |
| `italian_transliterations.txt` *(generated)* | Plain list of Italian transliterations, output of `dataset_creation.py`, used to train the target-side tokenizer. |
| `transliteration_dataset.jsonl` *(generated)* | Full parallel training/validation dataset, output of `dataset_creation.py`. |

---

## Pipeline

The scripts are meant to be run in this order:

```bash
# 1. Generate the parallel dataset from the CMU Pronouncing Dictionary
#    (uses cmudict.dict, which has test-set entries commented out
#    to prevent leakage — see "Data" below)
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
Reads `cmudict.dict`, filters out comments/variant markers (e.g. `word(2)`), strips numeric stress from the ARPABET phonemes, and converts each entry into an Italian transliteration via `mapping_cmu_italian.convert_cmu_to_ipt`. Produces:
- `transliteration_dataset.jsonl` — one JSON object per line: `english_word`, `cmu_with_stress`, `cmu_clean`, `italian_transliteration`.
- `english_words.txt` — plain list of English words, used to train the source-side tokenizers.
- `italian_transliterations.txt` — plain list of Italian transliterations, used to train the target-side tokenizer.

### 2. Phoneme-to-grapheme mapping — `mapping_cmu_italian.py`
`convert_cmu_to_ipt(cmu_phonemes: list[str]) -> str` applies:
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

## Running on Google Colab

All the code in this repository was actually orchestrated and run through a single Google Colab notebook (`Cross_Lingual_Phonetic_Translitteration.ipynb`), which wraps the pipeline above into GPU-backed cells: [colab.research.google.com/drive/1U2hLqS_U1mc61Toaph2MWMwwZC9Qkb-I](https://colab.research.google.com/drive/1U2hLqS_U1mc61Toaph2MWMwwZC9Qkb-I?usp=sharing)

> This link points to the notebook's original location on Google Drive, which may require explicit sharing/access permissions.

Its structure mirrors the pipeline one-to-one:

1. **Environment Setup and Repository Cloning** — clones this repo and installs `transformers`, `sentencepiece`, `jiwer`.
2. **Data Preprocessing and Dataset Generation** — runs `dataset_creation.py`.
3. **Training of BPE, Unigram, and Char tokenizers** — runs `tokenization.py`.
4. **Training Phase** — runs each of the four training scripts (`train_bart_bpe.py`, `train_bart_unigram.py`, `train_blstm_att_bpe.py`, `train_blstm_att_unigram.py`) in its own cell; after each run, the resulting checkpoint is zipped (for BART) and downloaded locally via `google.colab.files.download`.
5. **Model Weights Loading via Google Drive** — mounts Google Drive and copies/extracts previously saved checkpoints from `MyDrive/NLP/best_models` into the local Colab runtime, so training doesn't need to be repeated to run inference.
6. **Inference for Test Phase** — calls `run_inference(...)` from `inference.py` on a chosen checkpoint (e.g. `run_inference("blstm_att_unigram")`) to evaluate on the held-out test set.

To reproduce the results, open the notebook in Colab with a GPU runtime and run the cells top to bottom; steps 4 can be run selectively (only the configurations you need) since each training script is independent.

---

## Data

The repository includes two versions of the CMU Pronouncing Dictionary (v0.7b):

- **`cmudict.dict`** — the dictionary with the entries corresponding to the hold-out test set (`test.jsonl`) commented out. **This is the file `dataset_creation.py` should be pointed at**, so that test words (and their transliterations) never enter the training/validation split.
- **`cmudict_complete.dict`** — the original, unmodified dictionary, provided for reference/reproducibility.

The test set itself — 300 manually transcribed English words, together with their reference Italian transliterations — lives in `test.jsonl`, used to compute the final test CER (see [`inference.py`](#repository-structure)).

> **If you extend the test set with new words**, remember to comment out those entries in `cmudict.dict` (or otherwise remove them, together with any morphologically related neighbors) before re-running `dataset_creation.py`. Otherwise the new test words — or close variants of them — may leak into the training data, inflating the model's apparent performance on them.

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

Full methodology, error analysis, and discussion are available in the accompanying paper:
[Citro_Fornetti_HWP_report.pdf](Citro_Fornetti_HWP_report.pdf)

---

## Authors

- **Gaia Citro** — `citro.2026094@studenti.uniroma1.it`
- **Lucia Fornetti** — `fornetti.2214370@studenti.uniroma1.it`

Sapienza University of Rome — NLP 2025–2026
