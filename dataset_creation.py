
''' 
"""
Dataset creation module for English-to-Italian phonetic transliteration.
Reads the CMU Pronouncing Dictionary and generates a JSONL dataset.
"""

import os
import json
from typing import List, Dict

# Import the mapping dictionary from the local module
from mapping_cmu_italian import ARPABET_TO_ITALIAN_MAPPING

def get_italian_grapheme(arpabet_symbol: str) -> str:
    """
    Retrieves the corresponding Italian transliteration for a given ARPABET symbol.
    
    Args:
        arpabet_symbol (str): The CMU phonetic symbol (without stress numbers).
        
    Returns:
        str: The mapped Italian grapheme.
        
    Raises:
        KeyError: If the phoneme is not present in the mapping dictionary.
    """
    symbol_upper: str = arpabet_symbol.upper()
    if symbol_upper not in ARPABET_TO_ITALIAN_MAPPING:
        raise KeyError(f"ARPABET symbol '{symbol_upper}' not found in the mapping.")
    
    return ARPABET_TO_ITALIAN_MAPPING[symbol_upper]

def remove_stress_markers(phonemes: List[str]) -> List[str]:
    """
    Removes numerical stress markers from a list of CMU phonemes.
    Example: ['S', 'AH1', 'M'] -> ['S', 'AH', 'M']
    
    Args:
        phonemes (List[str]): List of original CMU phonemes containing numbers.
        
    Returns:
        List[str]: List of cleaned CMU phonemes with only alphabetic characters.
    """
    clean_phonemes: List[str] = []
    for phoneme in phonemes:
        # Filter out digits using list comprehension
        clean_phoneme: str = "".join([char for char in phoneme if not char.isdigit()])
        clean_phonemes.append(clean_phoneme)
    return clean_phonemes

def create_jsonl_dataset(input_file_path: str, output_file_path: str) -> None:
    """
    Processes the entire CMU dictionary and generates a JSONL dataset containing
    the English word, original CMU phonemes, clean CMU phonemes, and Italian transliteration.
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"The dictionary file was not found at: {input_file_path}")

    with open(input_file_path, "r", encoding="utf-8") as infile, \
         open(output_file_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            # Strip inline comments: split the string at '#' and keep only the first part
            if "#" in line:
                line = line.split("#")[0]
            
            # Skip comments that start with ';' or lines that are now empty after removing '#'
            if line.startswith(";") or not line.strip():
                continue
            
            components: List[str] = line.strip().split()
            english_word: str = components[0]
            
            # Original phonemes (with stress markers)
            original_phonemes_list: List[str] = components[1:]
            cmu_with_stress: str = " ".join(original_phonemes_list)
            
            # Clean phonemes (without stress markers)
            clean_phonemes_list: List[str] = remove_stress_markers(original_phonemes_list)
            cmu_clean: str = " ".join(clean_phonemes_list)
            
            # Italian transliteration mapping
            try:
                italian_chars: List[str] = [get_italian_grapheme(p) for p in clean_phonemes_list]
                italian_transliteration: str = "".join(italian_chars)
            except KeyError as error:
                print(f"Skipping word '{english_word}': {error}")
                continue
            
            # Construct the data dictionary for the current word
            dataset_entry: Dict[str, str] = {
                "english_word": english_word,
                "cmu_with_stress": cmu_with_stress,
                "cmu_clean": cmu_clean,
                "italian_transliteration": italian_transliteration
            }
            
            # Convert the dictionary to a JSON string and write it as a single line
            json_string: str = json.dumps(dataset_entry, ensure_ascii=False)
            outfile.write(json_string + "\n")

# --- Execution Entry Point ---
if __name__ == "__main__":
    # Define file paths
    dict_input_path: str = "cmudict.dict"
    dataset_output_path: str = "transliteration_dataset.jsonl"
    
    print(f"Starting dataset generation from '{dict_input_path}'...")
    
    try:
        create_jsonl_dataset(dict_input_path, dataset_output_path)
        print(f"Dataset successfully created and saved to: '{dataset_output_path}'")
    except Exception as e:
        print(f"An error occurred during dataset generation: {e}")
'''

import os
import json
from typing import List, Dict

# Import the mapping dictionary from the local module
from mapping_cmu_italian import ARPABET_TO_ITALIAN_MAPPING

def get_italian_grapheme(arpabet_symbol: str) -> str:
    # [Rimane identico al vostro codice originale]
    symbol_upper: str = arpabet_symbol.upper()
    if symbol_upper not in ARPABET_TO_ITALIAN_MAPPING:
        raise KeyError(f"ARPABET symbol '{symbol_upper}' not found in the mapping.")
    return ARPABET_TO_ITALIAN_MAPPING[symbol_upper]

def remove_stress_markers(phonemes: List[str]) -> List[str]:
    # [Rimane identico al vostro codice originale]
    clean_phonemes: List[str] = []
    for phoneme in phonemes:
        clean_phoneme: str = "".join([char for char in phoneme if not char.isdigit()])
        clean_phonemes.append(clean_phoneme)
    return clean_phonemes

def create_jsonl_dataset(input_file_path: str, output_file_path: str, 
                         english_txt_path: str, italian_txt_path: str) -> None:
    """
    Processes the CMU dictionary and generates:
    1. A JSONL dataset containing all information.
    2. A clean .txt file with only English words (for SentencePiece).
    3. A clean .txt file with only Italian transliterations (for SentencePiece).
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"The dictionary file was not found at: {input_file_path}")

    # Apriamo tutti e 3 i file contemporaneamente
    with open(input_file_path, "r", encoding="utf-8") as infile, \
         open(output_file_path, "w", encoding="utf-8") as outfile_jsonl, \
         open(english_txt_path, "w", encoding="utf-8") as outfile_en, \
         open(italian_txt_path, "w", encoding="utf-8") as outfile_it:
        
        for line in infile:
            if "#" in line:
                line = line.split("#")[0]
            
            if line.startswith(";") or not line.strip():
                continue
            
            components: List[str] = line.strip().split()
            english_word: str = components[0]
            
            original_phonemes_list: List[str] = components[1:]
            cmu_with_stress: str = " ".join(original_phonemes_list)
            
            clean_phonemes_list: List[str] = remove_stress_markers(original_phonemes_list)
            cmu_clean: str = " ".join(clean_phonemes_list)
            
            try:
                italian_chars: List[str] = [get_italian_grapheme(p) for p in clean_phonemes_list]
                italian_transliteration: str = "".join(italian_chars)
            except KeyError as error:
                print(f"Skipping word '{english_word}': {error}")
                continue
            
            # 1. Scriviamo il JSONL (come prima)
            dataset_entry: Dict[str, str] = {
                "english_word": english_word,
                "cmu_with_stress": cmu_with_stress,
                "cmu_clean": cmu_clean,
                "italian_transliteration": italian_transliteration
            }
            json_string: str = json.dumps(dataset_entry, ensure_ascii=False)
            outfile_jsonl.write(json_string + "\n")

            # 2. Scriviamo nei file TXT per SentencePiece
            outfile_en.write(english_word + "\n")
            outfile_it.write(italian_transliteration + "\n")

# --- Execution Entry Point ---
if __name__ == "__main__":
    dict_input_path: str = "cmudict.dict"
    dataset_output_path: str = "transliteration_dataset.jsonl"
    
    # Nuovi percorsi per i file TXT
    english_txt_path: str = "english_words.txt"
    italian_txt_path: str = "italian_transliterations.txt"
    
    print(f"Starting dataset generation from '{dict_input_path}'...")
    
    try:
        create_jsonl_dataset(dict_input_path, dataset_output_path, english_txt_path, italian_txt_path)
        print(f"Dataset JSONL created at: '{dataset_output_path}'")
        print(f"Testo Inglese salvato in: '{english_txt_path}'")
        print(f"Testo Italiano salvato in: '{italian_txt_path}'")
    except Exception as e:
        print(f"An error occurred during dataset generation: {e}")