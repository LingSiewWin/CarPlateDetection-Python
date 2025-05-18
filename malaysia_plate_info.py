# malaysia_plate_info.py

import re
import difflib

def preprocess_ocr_text(text):
    text = text.upper().replace(' ', '').replace('-', '')
    corrections = {
        '0': 'O', '1': 'I', '8': 'B', '5': 'S', '6': 'G', '2': 'Z', 'Q': '0'
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text


def identify_state(text):
    text = preprocess_ocr_text(text)

    peninsular_states = {
        'A': ('Perak', 'Private'),
        'B': ('Selangor', 'Private'),
        'C': ('Pahang', 'Private'),
        'D': ('Kelantan', 'Private'),
        'J': ('Johor', 'Private'),
        'K': ('Kedah', 'Private'),
        'L': ('Labuan', 'Private'),
        'M': ('Melaka', 'Private'),
        'N': ('Negeri Sembilan', 'Private'),
        'P': ('Penang', 'Private'),
        'R': ('Perlis', 'Private'),
        'T': ('Terengganu', 'Private'),
        'V': ('Kuala Lumpur', 'Private'),
        'W': ('Kuala Lumpur', 'Private'),
        'F': ('Putrajaya', 'Private'),
        'KV': ('Langkawi', 'Private')
    }

    sarawak_divisions = {
        'QA': ('Kuching', 'Private'),
        'QB': ('Sri Aman/Betong', 'Private'),
        'QC': ('Samarahan/Serian', 'Private'),
        'QD': ('Bintulu', 'Private'),
        'QL': ('Limbang', 'Private'),
        'QM': ('Miri', 'Private'),
        'QP': ('Kapit', 'Private'),
        'QR': ('Sarikei', 'Private'),
        'QS': ('Sibu/Mukah', 'Private'),
        'QSG': ('Sarawak Government', 'Government')
    }

    sabah_divisions = {
        'SA': ('West Coast', 'Private'),
        'SB': ('Beaufort', 'Private'),
        'SD': ('Lahad Datu', 'Private'),
        'SJ': ('West Coast', 'Private'),
        'SK': ('Kudat', 'Private'),
        'SM': ('Sandakan', 'Private'),
        'SS': ('Sandakan', 'Private'),
        'ST': ('Tawau', 'Private'),
        'SU': ('Keningau', 'Private'),
        'SW': ('Tawau', 'Private'),
        'SG': ('Sabah Government', 'Government'),
        'SMJ': ('Sabah Government', 'Government')
    }

    special_plates = {
        'TAXI': ('Taxi', 'Commercial'),
        'CC': ('Diplomatic Corps', 'Diplomatic'),
        'DC': ('Diplomatic Corps', 'Diplomatic'),
        'UN': ('United Nations', 'Diplomatic'),
        'PA': ('International Organization', 'Diplomatic'),
        'ZA': ('Malaysian Army', 'Military'),
        'ZB': ('Malaysian Army', 'Military'),
        'ZC': ('Malaysian Army', 'Military'),
        'ZD': ('Malaysian Army', 'Military'),
        'ZL': ('Royal Malaysian Navy', 'Military'),
        'ZU': ('Royal Malaysian Air Force', 'Military'),
        'ZZ': ('Ministry of Defence', 'Military'),
        'G': ('Government', 'Government'),
        'LIMO': ('KLIA Limousine', 'Commercial')
    }

    all_prefixes = (
        list(peninsular_states.keys()) +
        list(sarawak_divisions.keys()) +
        list(sabah_divisions.keys()) +
        list(special_plates.keys())
    )

    for i in range(2, min(5, len(text)) + 1):
        prefix = text[:i]
        matches = difflib.get_close_matches(prefix, all_prefixes, n=1, cutoff=0.8)
        if matches:
            prefix = matches[0]
            if prefix in special_plates:
                return special_plates[prefix]
            elif prefix in sarawak_divisions:
                return sarawak_divisions[prefix]
            elif prefix in sabah_divisions:
                return sabah_divisions[prefix]
            elif prefix in peninsular_states:
                return peninsular_states[prefix]

    special_patterns = {
        r'CC\d+': ('Diplomatic Corps', 'Diplomatic'),
        r'\d+-\d+-UN': ('United Nations', 'Diplomatic'),
        r'\d+-\d+-PA': ('International Organization', 'Diplomatic'),
        r'Z[A-Z]\d+': ('Malaysian Armed Forces', 'Military'),
        r'ZZ\d+': ('Ministry of Defence', 'Military')
    }

    for pattern, result in special_patterns.items():
        if re.search(pattern, text):
            return result

    return "Unknown", "Unknown"