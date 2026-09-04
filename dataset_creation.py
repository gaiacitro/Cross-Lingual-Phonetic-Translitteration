import os
import json
from typing import List, Dict

# 1. Importing the mapping function from mapping_cmu_italian.py
from mapping_cmu_italian import convert_cmu_to_tfi

# 2. Function to remove stress markers from CMU phonemes
def remove_stress_markers(phonemes: List[str]) -> List[str]:
    clean_phonemes: List[str] = []
    for phoneme in phonemes:
        clean_phoneme: str = "".join([char for char in phoneme if not char.isdigit()])
        clean_phonemes.append(clean_phoneme)
    return clean_phonemes

# 3. Creation pipeline for dataset
def create_jsonl_dataset(input_file_path: str, output_file_path: str, 
                         english_txt_path: str, italian_txt_path: str) -> None:
    '''
    Process the CMU dictionary and generate:
    1. A JSONL dataset with all pairs and metadata.
    2. A clean .txt file with English words (for SentencePiece).
    3. A clean .txt file with Italian transliterations (for SentencePiece).
    '''
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"The dictionary file was not found at: {input_file_path}")

    # Open the input and output files
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
            english_word: str = components[0].split("(")[0]
            
            original_phonemes_list: List[str] = components[1:]
            cmu_with_stress: str = " ".join(original_phonemes_list)
            
            clean_phonemes_list: List[str] = remove_stress_markers(original_phonemes_list)
            cmu_clean: str = " ".join(clean_phonemes_list)
            
            try:
                # Pass the entire list to the function to allow lookahead
                italian_transliteration: str = convert_cmu_to_tfi(clean_phonemes_list)

                # Security check to ensure the function does not return an empty string due to unexpected errors
                if not italian_transliteration:
                    raise ValueError("Empty transliteration generated.")
                    
            except Exception as error:
                print(f"Skipping word '{english_word}': {error}")
                continue
            
            # Save the dataset entry in JSONL format
            dataset_entry: Dict[str, str] = {
                "english_word": english_word,
                "cmu_with_stress": cmu_with_stress,
                "cmu_clean": cmu_clean,
                "italian_transliteration": italian_transliteration
            }
            json_string: str = json.dumps(dataset_entry, ensure_ascii=False)
            outfile_jsonl.write(json_string + "\n")

            # Save the text files for SentencePiece
            outfile_en.write(english_word + "\n")
            outfile_it.write(italian_transliteration + "\n")

# 4. Entry point
if __name__ == "__main__":
    dict_input_path: str = "cmudict.dict"
    dataset_output_path: str = "transliteration_dataset.jsonl"
    
    english_txt_path: str = "english_words.txt"
    italian_txt_path: str = "italian_transliterations.txt"
    
    print(f"Starting dataset generation from '{dict_input_path}'...")
    
    try:
        create_jsonl_dataset(dict_input_path, dataset_output_path, english_txt_path, italian_txt_path)
        print(f"Dataset JSONL created at: '{dataset_output_path}'")
        print(f"English vocabulary saved to: '{english_txt_path}'")
        print(f"Italian transliterations saved to: '{italian_txt_path}'")
    except Exception as e:
        print(f"An error occurred during dataset generation: {e}")