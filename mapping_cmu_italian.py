def convert_cmu_to_tfi(cmu_phonemes):
    front_vowels = {"EH", "EY", "IH", "IY", "ER"}

    base_mapping = {
        "AA": "a", 
        "AE": "a", 
        "AH": "a", 
        "AO": "o", 
        "AW": "au", 
        "AY": "ai",
        "EH": "e", 
        "ER": "er", 
        "EY": "ei", 
        "IH": "i", 
        "IY": "i", 
        "OW": "ou",
        "OY": "oi", 
        "UH": "u", 
        "UW": "u", 
        "B": "b", 
        "D": "d", 
        "DH": "d",
        "F": "f", 
        "HH": "h", 
        "L": "l", 
        "M": "m", 
        "N": "n", 
        "NG": "ngh", 
        "P": "p", 
        "R": "r", 
        "S": "s", 
        "SH": "sci", 
        "T": "t", 
        "TH": "t", 
        "V": "v", 
        "W": "u", 
        "Y": "i", 
        "Z": "z", 
        "ZH": "sci"
    }

    tfi_chars = []
    
    for i, phoneme in enumerate(cmu_phonemes):
        p_clean = ''.join([c for c in phoneme if not c.isdigit()])

        next_p = None
        if i + 1 < len(cmu_phonemes):
            next_p = ''.join([c for c in cmu_phonemes[i+1] if not c.isdigit()])

        # --- 1. CH (Soft C: c / ci) ---
        if p_clean == "CH":
            if next_p in front_vowels:
                tfi_chars.append("c")    # E.g.: CH + EH -> ce
            elif next_p is None:
                tfi_chars.append("c")    # End of word -> c (e.g. arc)
            else:
                tfi_chars.append("ci")   # Before a/o/u or a consonant -> ci

        # --- 2. JH (Soft G: g / gi) ---
        elif p_clean == "JH":
            if next_p in front_vowels:
                tfi_chars.append("g")    # E.g.: JH + EH -> ge
            elif next_p is None:
                tfi_chars.append("g")    # End of word -> g (e.g. frig)
            else:
                tfi_chars.append("gi")   # Before a/o/u or a consonant -> gi

        # --- 3. K (Hard C: ch / c) ---
        elif p_clean == "K":
            if next_p in front_vowels:
                tfi_chars.append("ch")   # E.g.: K + IY -> chi
            elif next_p is None:
                tfi_chars.append("ch")   # End of word -> ch (e.g. darch)
            else:
                tfi_chars.append("c")    # Before a/o/u or a consonant -> c (e.g. cloud -> claud)

        # --- 4. G (Hard G: gh / g) ---
        elif p_clean == "G":
            if next_p in front_vowels:
                tfi_chars.append("gh")   # E.g.: G + EY -> ghei
            elif next_p is None:
                tfi_chars.append("gh")   # End of word -> gh (e.g. bagh)
            else:
                tfi_chars.append("g")    # Before a/o/u or a consonant -> g (e.g. green -> grin)

        # --- Mapping standard ---
        else:
            tfi_chars.append(base_mapping.get(p_clean, ""))

    return "".join(tfi_chars)

# Behavior examples:
# "fridge" -> ['F', 'R', 'IH1', 'JH'] -> "frig"
# "arch"   -> ['AA1', 'R', 'CH'] -> "arc"
# "cloud"  -> ['K', 'L', 'AW1', 'D'] -> "claud"
# "king"   -> ['K', 'IH1', 'NG'] -> "chingh"