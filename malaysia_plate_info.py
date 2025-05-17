import re

def identify_state(text):
    """
    Identifies Malaysian state or plate type from license plate text.
    Parameters:
        text (str): License plate number string
    Returns:
        tuple: (state, plate_type)
    """
    # Normalize input
    text = text.upper().replace('-', '').replace(' ', '')
    # Peninsular Malaysia prefixes (single/double letter)
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
    # Sarawak divisions (QA, QB, QC...)
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
    # Sabah divisions (SA, SB, SD...)
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
    # Special plates
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
    # Step 1: Check for special plate formats
    for key, value in special_plates.items():
        if key in text:
            return value[0], value[1]
    # Use regex to match patterns like CC1234 or 12-34-UN
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
    # Step 2: Check Sarawak and Sabah division prefixes
    if len(text) >= 2:
        prefix = text[:2]
        if prefix in sarawak_divisions:
            return sarawak_divisions[prefix][0], sarawak_divisions[prefix][1]
        elif prefix in sabah_divisions:
            return sabah_divisions[prefix][0], sabah_divisions[prefix][1]
    # Step 3: Check Peninsular Malaysia states
    for i in range(min(2, len(text)), 0, -1):
        if text[:i] in peninsular_states:
            return peninsular_states[text[:i]][0], peninsular_states[text[:i]][1]
    # Default unknown
    return "Unknown", "Unknown" 