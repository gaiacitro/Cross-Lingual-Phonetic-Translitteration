from typing import Dict

ARPABET_TO_ITALIAN_MAPPING: Dict[str, str] = {
    # --- Vowels ---
    "AA": "a",   # e.g., 'odd' -> a
    "AE": "a",   # e.g., 'at' -> a
    "AH": "a",   # e.g., 'hut' -> a
    "AO": "o",   # e.g., 'ought' -> o
    "AW": "au",  # e.g., 'cow' -> au
    "AY": "ai",  # e.g., 'hide' -> ai
    "EH": "e",   # e.g., 'Ed' -> e
    "ER": "er",  # e.g., 'hurt' -> er (rhotic vowel)
    "EY": "ei",  # e.g., 'ate' -> ei
    "IH": "i",   # e.g., 'it' -> i
    "IY": "i",   # e.g., 'eat' -> i
    "OW": "ou",  # e.g., 'oat' -> ou
    "OY": "oi",  # e.g., 'toy' -> oi
    "UH": "u",   # e.g., 'hood' -> u
    "UW": "u",   # e.g., 'two' -> u

    # --- Consonants ---
    "B": "b",    # e.g., 'be' -> b
    "CH": "ci",  # e.g., 'cheese' -> ci (soft 'c' sound in Italian)
    "D": "d",    # e.g., 'dee' -> d
    "DH": "d",   # e.g., 'thee' -> d (Italian lacks dental fricatives, mapped to 'd')
    "F": "f",    # e.g., 'fee' -> f
    "G": "g",    # e.g., 'green' -> g (hard 'g' sound)
    "HH": "h",   # e.g., 'he' -> h (often silent in Italian, but kept for model pattern learning)
    "JH": "gi",  # e.g., 'gee' -> gi (soft 'g' sound in Italian)
    "K": "k",    # e.g., 'key' -> k (unambiguous hard sound compared to Italian 'c'/'ch')
    "L": "l",    # e.g., 'lee' -> l
    "M": "m",    # e.g., 'me' -> m
    "N": "n",    # e.g., 'knee' -> n
    "NG": "ng",  # e.g., 'ping' -> ng
    "P": "p",    # e.g., 'pee' -> p
    "R": "r",    # e.g., 'read' -> r
    "S": "s",    # e.g., 'sea' -> s
    "SH": "sci", # e.g., 'she' -> sci (unambiguous soft 'sc' sound in Italian)
    "T": "t",    # e.g., 'tea' -> t
    "TH": "t",   # e.g., 'theta' -> t (unvoiced dental fricative mapped to 't')
    "V": "v",    # e.g., 'vee' -> v
    "W": "u",    # e.g., 'we' -> u (semivowel mapped to 'u')
    "Y": "i",    # e.g., 'yield' -> i (semivowel mapped to 'i')
    "Z": "z",    # e.g., 'zee' -> z
    "ZH": "j"    # e.g., 'seizure' -> j (approximate mapping for voiced postalveolar fricative)
}