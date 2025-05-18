# malaysia_plate_info.py

import re
import difflib

def preprocess_ocr_text(text):
    """
    Preprocess OCR text with common corrections and format enforcement.
    """
    text = text.upper().replace(' ', '').replace('-', '')
    
    # Only correct the first 1-3 chars (prefix)
    prefix_raw = text[:3]
    number_raw = text[3:]

    # Corrections for prefix only (letters only)
    prefix_corrections = {
        '0': 'O', '1': 'I', '8': 'B', '5': 'S', '6': 'G', '2': 'Z', 'Q': 'O',
        '13': 'B'
    }
    for wrong, right in prefix_corrections.items():
        prefix_raw = prefix_raw.replace(wrong, right)
    prefix = ''.join(c for c in prefix_raw if c.isalpha())

    # For the number part, convert common OCR letter errors to digits
    number_corrections = {
        'S': '5', 'B': '8', 'I': '1', 'O': '0', 'Z': '2', 'G': '6'
    }
    numbers = ''
    for c in number_raw:
        if c.isdigit():
            numbers += c
        elif c in number_corrections:
            numbers += number_corrections[c]
        # else: skip any other letters

    return prefix + numbers

def parse_plate_format(text):
    """
    Strictly parse and validate Malaysian plate format.
    Returns (prefix, numbers, is_valid)
    """
    text = preprocess_ocr_text(text)
    
    # Malaysian plate format: 1-3 letters followed by 1-4 numbers
    pattern = r'^([A-Z]{1,3})(\d{1,4})$'
    match = re.match(pattern, text)
    
    if match:
        prefix, numbers = match.groups()
        return prefix, numbers, True
    return None, None, False

def identify_state(text):
    """
    Identify state and plate type from the plate text.
    """
    text = preprocess_ocr_text(text)
    prefix, numbers, is_valid = parse_plate_format(text)
    
    if not is_valid:
        return "Invalid Format", "Unknown"

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

    # Simplified Sabah and Sarawak divisions
    sarawak_prefixes = ['QA', 'QB', 'QC', 'QD', 'QL', 'QM', 'QP', 'QR', 'QS', 'QSG']
    sabah_prefixes = ['SA', 'SB', 'SD', 'SJ', 'SK', 'SM', 'SS', 'ST', 'SU', 'SW', 'SG', 'SMJ']

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
    }

    # Check for special plates first
    for i in range(2, min(5, len(prefix)) + 1):
        test_prefix = prefix[:i]
        if test_prefix in special_plates:
            return special_plates[test_prefix]

    # Check for Sarawak
    if any(prefix.startswith(p) for p in sarawak_prefixes):
        return ('Sarawak', 'Private')

    # Check for Sabah
    if any(prefix.startswith(p) for p in sabah_prefixes):
        return ('Sabah', 'Private')

    # Check peninsular states using only the first letter
    if prefix and prefix[0] in peninsular_states:
        return peninsular_states[prefix[0]]

    # Special patterns for diplomatic and military plates
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