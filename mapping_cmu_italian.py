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

        # --- 1. CH (C Morbida: c / ci) ---
        if p_clean == "CH":
            if next_p in front_vowels:
                tfi_chars.append("c")    # Es: CH + EH -> ce
            elif next_p is None:
                tfi_chars.append("c")    # Fine parola -> c (es. arc)
            else:
                tfi_chars.append("ci")   # Davanti ad a/o/u o cons. -> ci

        # --- 2. JH (G Morbida: g / gi) ---
        elif p_clean == "JH":
            if next_p in front_vowels:
                tfi_chars.append("g")    # Es: JH + EH -> ge
            elif next_p is None:
                tfi_chars.append("g")    # Fine parola -> g (es. frig)
            else:
                tfi_chars.append("gi")   # Davanti ad a/o/u o cons. -> gi

        # --- 3. K (C Dura: ch / c) ---
        elif p_clean == "K":
            if next_p in front_vowels:
                tfi_chars.append("ch")   # Es: K + IY -> chi
            elif next_p is None:
                tfi_chars.append("ch")   # Fine parola -> ch (es. darch)
            else:
                tfi_chars.append("c")    # Davanti ad a/o/u o cons. -> c (es. cloud -> claud)

        # --- 4. G (G Dura: gh / g) ---
        elif p_clean == "G":
            if next_p in front_vowels:
                tfi_chars.append("gh")   # Es: G + EY -> ghei
            elif next_p is None:
                tfi_chars.append("gh")   # Fine parola -> gh (es. bagh)
            else:
                tfi_chars.append("g")    # Davanti ad a/o/u o cons. -> g (es. green -> grin)

        # --- Mapping standard ---
        else:
            tfi_chars.append(base_mapping.get(p_clean, ""))

    return "".join(tfi_chars)

# Esempi di comportamento:
# "fridge" -> ['F', 'R', 'IH1', 'JH'] -> "frig"
# "arch"   -> ['AA1', 'R', 'CH'] -> "arc"
# "cloud"  -> ['K', 'L', 'AW1', 'D'] -> "claud"
# "king"   -> ['K', 'IH1', 'NG'] -> "chingh"