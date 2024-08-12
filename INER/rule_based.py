import nltk
import re
# nltk.download("punkt")
from nltk.tokenize import word_tokenize

import pprint

# Features Dictionary List
# ============================

PPRE = "PPRE"
PMID = "PMID"
PSUF = "PSUF"
PTIT = "PTIT"
OPRE = "OPRE"
OSUF = "OSUF"
OPOS = "OPOS"
OCON = "OCON"
LPRE = "LPRE"
LSUF = "LSUF"
LLDR = "LLDR"
POLP = "POLP"
LOPP = "LOPP"
DISASTER = "DISASTER"
TIME = "TIME"
DAY = "DAY"
MONTH = "MONTH"
# 1. Contextual Features Dictionary
contextual_features_dictionary = {
    "PPRE": ["Dr.", "Pak", "K.H.", "bin", "van", "SKom", "SH"],
    "PMID": ["bin", "van","binti"],
    "PSUF": ["SKom", "SH"],
    "PTIT": ["Menristek", "Mendagri"],
    "OPRE": ["PT.", "Universitas"],
    "OSUF": ["Ltd.", "Company"],
    "OPOS": ["Ketua"],
    "OCON": ["Muktamar", "Rakernas"],
    "LPRE": ["Kota", "Provinsi","Kabupaten","Kecamatan","Kelurahan","Desa","kota","provinsi","kabupaten","kecamatan","kelurahan","desa"],
    "LSUF": ["Utara","Selatan","Barat","Tengah","Timur","City","utara","selatan","barat","tengah","timur","Nugini"],
    "LLDR": ["Gubernur", "Walikota"],
    "POLP": ["oleh", "untuk"],
    "LOPP": ["di", "ke", "dari"],
    # IMPACT TODO ini belum dimasukin ke rule
    "DISASTER": ["banjir", "gempa","bumi", "tsunami","gunung", "meletus", "tanah","longsor", "kekeringan","angin","topan","Banjir","Gempa","Bumi","Tsunami","Gunung","Meletus","Tanah","Longsor","Kekeringan","Angin","Topan"],
    "IMPACT": ["korban jiwa", "rusak berat", "terdampak", "mengungsi", "kerugian", "hilang", "terluka", "mati", "hancur", "terendam"],
    "TIME": ["hari ini", "kemarin", "minggu ini", "bulan ini", "tahun ini", "pagi", "siang", "sore", "malam"],
    "DAY": ["Senin", "Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu","senin", "selasa","rabu","kamis","jumat","sabtu","minggu"],
    "MONTH": ["Januari","Februari","Maret","April", "Mei","Juni","Juli","Agustus","September","Oktober","November","Desember","januari","februari","maret","april", "mei","juni","juli","agustus","september","oktober","november","desember"]
}

# 2. Morphological Features Dictionary
TITLE_CASE = "TitleCase"
UPPER_CASE = 'UpperCase'
LOWER_CASE = "LowerCase"
MIXED_CASE = "MixedCase"
CAP_START = 'CapStart'
CHAR_DIGIT = 'CharDigit'
DIGIT = 'Digit'
NUMERIC = "Numeric"
NUMSTR = "NumStr"
ROMAN = "Roman"
TIME_FORM = "TimeForm"
morphological_features_dictionary = {
    'TitleCase': r'^[A-Z][a-z]+$',
    'UpperCase': r'^[A-Z]+$',
    'LowerCase': r'^[a-z]+$',
    'MixedCase': r'^(?=.*[a-z])(?=.*[A-Z])[A-Za-z]+$',
    'CapStart': r'^[A-Z]',
    'CharDigit': r'^(?=.*[a-zA-Z])(?=.*[0-9])[A-Za-z0-9]+$',
    'Digit': r'^\d+$',
    'DigitSlash': r'^\d+/\d+$',
    'Numeric': r'^\d+[\.,]\d+$',
    'NumStr': None,  # This would need a predefined list or more complex natural language processing.
    'Roman': r'^M{0,4}(CM|CD|XC|XL|IX|IV|I{1,3}|V?I{0,3})$',
    'TimeForm': r'^\d{1,2}[:.]\d{2}$'
}

# 3. Part Of Speech Features Dictionary
ART = "ART"
ADJ = 'ADJ'
ADV = 'ADV'
AUX = 'AUX'
C = 'C'
DEF = 'DEF'
NOUN = 'NOUN'
NOUNP = 'NOUNP'
NUM = 'NUM'
MODAL = 'MODAL'
PAR = 'PAR'
PREP = 'PREP'
PRO = 'PRO'
VACT = 'VACT'
VPAS = 'VPAS'
VERB = 'VERB'
OOV = 'OOV'
partOfSpeech_features_dictionary = {
    "si": "ART", "sang": "ART", "seorang": "ART",
    "indah": "ADJ", "baik": "ADJ", "luar biasa": "ADJ", "buruk": "ADJ", "tinggi": "ADJ",
    "telah": "ADV", "kemarin": "ADV", "segera": "ADV", "baru": "ADV", "nanti": "ADV",
    "harus": "AUX", "bisa": "AUX", "boleh": "AUX",
    "dan": "C", "atau": "C", "lalu": "C", "tetapi": "C", "namun": "C",
    "merupakan": "DEF", "adalah": "DEF", "ialah": "DEF",
    "rumah": "NOUN", "gedung": "NOUN", "kantor": "NOUN", "sekolah": "NOUN", "jalan": "NOUN",
    "ayah": "NOUNP", "ibu": "NOUNP", "guru": "NOUNP", "dokter": "NOUNP", "polisi": "NOUNP",
    "satu": "NUM", "dua": "NUM", "tiga": "NUM", "empat": "NUM", "lima": "NUM",
    "akan": "MODAL", "mungkin": "MODAL", "pasti": "MODAL", "dapat": "MODAL",
    "kah": "PAR", "pun": "PAR", "lah": "PAR", "dong": "PAR",
    "di": "PREP", "ke": "PREP", "dari": "PREP", "atas": "PREP", "dengan": "PREP",
    "saya": "PRO", "beliau": "PRO", "kami": "PRO", "kita": "PRO", "mereka": "PRO",
    "menuduh": "VACT", "membaca": "VACT", "menulis": "VACT",
    "dituduh": "VPAS", "dibaca": "VPAS", "ditulis": "VPAS",
    "pergi": "VERB", "tidur": "VERB", "makan": "VERB", "minum": "VERB", "datang": "VERB"
}
# Function to classify tokens
def classify_token(token):
    # Number Checker
    if re.match("^\d+\.?\d*$", token):
        return (token, "NUM")
    # Punctuation Checker
    elif re.match(r'^\W+$', token):  # This checks for any non-alphanumeric character
        if token in ['(', '{', '[']:  # OPUNC Checker
            return (token, "OPUNC")
        elif token in [')', '}', ']']:  # EPUNC Checker
            return (token, "EPUNC")
        else:
            return (token, "OPUNC")
    else:
        return (token, "WORD")
    



# =======================================
# 1. Contextual Features Assignment
def classify_contextual_features(tokens, features):
  classified_contextual_features = []
  for token in tokens:
    found = False
    for feature, values in features.items():
      if token[0] in values:
        classified_contextual_features.append((token[0],token[1],feature))
        found = True
    if not found:
      classified_contextual_features.append((token[0],token[1],""))
  return classified_contextual_features

# 2. Morphological Featrues Assignment
def classify_morphological_features(tokens,features):
    classified_morphological_features = []
    # NumStr Dictionary
    num_str_words = ['satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh']

    # Check each pattern
    for token in tokens:
      found = False
      for feature, pattern in features.items():
        if found:
          continue
        if pattern and re.fullmatch(pattern, token[0]):
            classified_morphological_features.append((token[0],token[1],token[2],feature))
            found = True

      # Check NumStr by list
      if token[0].lower() in num_str_words:
         classified_morphological_features.append((token[0],token[1],token[2],"NumStr"))
         found = True

      # not found
      if not found:
        classified_morphological_features.append((token[0],token[1],token[2],"Unknown"))

    return classified_morphological_features

# 3. Part Of Speech Feature Assignment
def classify_partOfSpeech_features(tokens,features):
  classified_partOfSpeech_features = []

  for token in tokens:
    classified_partOfSpeech_features.append((token[0],token[1],token[2],token[3],partOfSpeech_features_dictionary.get(token[0],"OOV")))

  return classified_partOfSpeech_features

def match_lokasi_bencana(tokens):
    result_lokasi_bencana = []
    pengurang = 0
    if len(tokens) <= 3:
      pengurang = 0
    else:
      pengurang = 3
    for i in range(len(tokens)):
      try:
        # Rules 1
        if ((tokens[i][1] == 'WORD' and tokens[i][2] == LOPP and tokens[i][3] == LOWER_CASE and tokens[i][4] == PREP) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV)):
            # print(f"Lokasi Bencana: {' '.join([tokens[i+1][0]])}")
            result_lokasi_bencana.append(tokens[i+1][0])

        # Rules 2
        if ((tokens[i][1] == 'WORD' and tokens[i][2] == LOPP and tokens[i][3] == LOWER_CASE and tokens[i][4] == PREP) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV) and
            (tokens[i+2][1] == "WORD" and  tokens[i+2][2] == LSUF and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV)):
            # print(f"Lokasi Bencana: {' '.join([tokens[i+1][0], tokens[i+2][0]])}")
            result_lokasi_bencana.append(tokens[i+1][0])
            result_lokasi_bencana.append(tokens[i+2][0])

        # Rules 3
        if ((tokens[i][1] == 'WORD' and tokens[i][2] == LOPP and tokens[i][3] == LOWER_CASE and tokens[i][4] == PREP) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == LPRE and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV) and
            (tokens[i+2][1] == "WORD" and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV)):
            # print(f"Lokasi Bencana: {' '.join([tokens[i+1][0], tokens[i+2][0]])}")
            result_lokasi_bencana.append(tokens[i+1][0])
            result_lokasi_bencana.append(tokens[i+2][0])

        # Rules 4
        if ((tokens[i][1] == 'WORD' and tokens[i][2] == LOPP and tokens[i][3] == LOWER_CASE and tokens[i][4] == PREP) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == LPRE and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV) and
            (tokens[i+2][1] == "WORD" and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV) and
            (tokens[i+3][1] == "WORD" and tokens[i+3][2] == LSUF and tokens[i+3][3] == TITLE_CASE and tokens[i+3][4] == OOV)):
            # print(f"Lokasi Bencana: {' '.join([tokens[i+2][0]], tokens[i+3][0])}")
            result_lokasi_bencana.append(tokens[i+2][0])
            result_lokasi_bencana.append(tokens[i+3][0])

        # Rules 5
        if ((tokens[i][1] == 'WORD' and tokens[i][2] == LPRE and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV)):
            # print(f"Lokasi Bencana: {' '.join([tokens[i+1][0]])}")
            result_lokasi_bencana.append(tokens[i+1][0])

        # Rules 6
        if ((tokens[i][1] == 'WORD' and tokens[i][2] == LOPP and tokens[i][3] == LOWER_CASE and tokens[i][4] == PREP) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV)):
            # print(f"Lokasi Bencana: {' '.join([tokens[i+1][0]])}")
            result_lokasi_bencana.append(tokens[i+1][0])

        # Rules 7
        if ((tokens[i][1] == 'WORD' and  tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
            (tokens[i+1][1] == 'WORD' and tokens[i][2] == LSUF and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV)):
            # print(f"Lokasi Bencana: {' '.join([tokens[i][0], tokens[i+1][0]])}")
            result_lokasi_bencana.append(tokens[i][0])
            result_lokasi_bencana.append(tokens[i+1][0])
      except IndexError:
        index_error = 1
    return set(result_lokasi_bencana)


def match_dampak_bencana(tokens):
    result_dampak_bencana = []
    pengurang = 0
    if len(tokens) <= 4:
      pengurang = 0
    else:
      pengurang = 4
    for i in range(len(tokens)):
      try:
        # Rules 1
        if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV ) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV) and
            (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == DIGIT and tokens[i+2][4] == OOV ) and
            (tokens[i+3][1] == 'WORD' and tokens[i+3][3] == LOWER_CASE and tokens[i+3][4] == NOUN ) and
            (tokens[i+4][1] == 'WORD' and tokens[i+4][3] == LOWER_CASE and tokens[i+4][4] == OOV)):
            # print(f"Dampak Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0], tokens[i+3][0], tokens[i+4][0]])}")
            result_dampak_bencana.append(tokens[i][0])
            result_dampak_bencana.append(tokens[i+1][0])
            result_dampak_bencana.append(tokens[i+2][0])
            result_dampak_bencana.append(tokens[i+3][0])
            result_dampak_bencana.append(tokens[i+4][0])

        # Rules 2
        if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV ) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV) and
            (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == DIGIT and tokens[i+2][4] == OOV ) and
            (tokens[i+3][1] == 'WORD' and tokens[i+3][3] == LOWER_CASE and tokens[i+3][4] == NOUN ) and
            (tokens[i+4][1] == 'WORD' and tokens[i+4][3] == LOWER_CASE and tokens[i+4][4] == OOV)):
            # print(f"Dampak Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0], tokens[i+3][0], tokens[i+4][0]])}")
            result_dampak_bencana.append(tokens[i][0])
            result_dampak_bencana.append(tokens[i+1][0])
            result_dampak_bencana.append(tokens[i+2][0])
            result_dampak_bencana.append(tokens[i+3][0])
            result_dampak_bencana.append(tokens[i+4][0])

        # Rules 3
        if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV ) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == DIGIT and tokens[i+1][4] == OOV) and
            (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == LOWER_CASE and tokens[i+2][4] == NOUN ) and
            (tokens[i+3][1] == 'WORD' and tokens[i+3][3] == LOWER_CASE and tokens[i+3][4] == OOV )):
            # print(f"Dampak Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0], tokens[i+3][0], tokens[i+4][0]])}")
            result_dampak_bencana.append(tokens[i][0])
            result_dampak_bencana.append(tokens[i+1][0])
            result_dampak_bencana.append(tokens[i+2][0])
            result_dampak_bencana.append(tokens[i+3][0])
            result_dampak_bencana.append(tokens[i+4][0])

        # Rules 4
        if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV ) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == DIGIT and tokens[i+1][4] == OOV) and
            (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == LOWER_CASE and tokens[i+2][4] == NOUN ) and
            (tokens[i+3][1] == 'WORD' and tokens[i+3][3] == LOWER_CASE and tokens[i+3][4] == OOV )):
            # print(f"Dampak Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0], tokens[i+3][0], tokens[i+4][0]])}")
            result_dampak_bencana.append(tokens[i][0])
            result_dampak_bencana.append(tokens[i+1][0])
            result_dampak_bencana.append(tokens[i+2][0])
            result_dampak_bencana.append(tokens[i+3][0])
            result_dampak_bencana.append(tokens[i+4][0])

        # Rules 5
        if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == NOUN ) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV) and
            (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == LOWER_CASE and tokens[i+2][4] == OOV ) and
            (tokens[i+3][1] == 'WORD' and tokens[i+3][3] == LOWER_CASE and tokens[i+3][4] == OOV ) and
            (tokens[i+4][1] == 'WORD' and tokens[i+4][3] == LOWER_CASE and tokens[i+4][4] == OOV)):
            # print(f"Dampak Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0], tokens[i+3][0], tokens[i+4][0]])}")
            result_dampak_bencana.append(tokens[i][0])
            result_dampak_bencana.append(tokens[i+1][0])
            result_dampak_bencana.append(tokens[i+2][0])
            result_dampak_bencana.append(tokens[i+3][0])
            result_dampak_bencana.append(tokens[i+4][0])

      except IndexError:
       index_error = 1
    return set(result_dampak_bencana)


def match_jenis_bencana(tokens):
    result_jenis_bencana = []
    pengurang = 0
    if len(tokens) <= 2:
      pengurang = 0
    else:
      pengurang = 2
    for i in range(len(tokens)):
        try:
          # Rules 1
          if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
              (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == DISASTER and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV) and
              (tokens[i+2][1] == 'WORD' and tokens[i+2][2] == DISASTER and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV)):
              # print(f"Jenis Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
              result_jenis_bencana.append(tokens[i][0])
              result_jenis_bencana.append(tokens[i+1][0])
              result_jenis_bencana.append(tokens[i+2][0])

          # Rules 2
          if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
              (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == DISASTER and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV) and
              (tokens[i+2][1] == 'WORD' and tokens[i+2][2] == DISASTER and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV)):
              # print(f"Jenis Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
              result_jenis_bencana.append(tokens[i][0])
              result_jenis_bencana.append(tokens[i+1][0])
              result_jenis_bencana.append(tokens[i+2][0])

          # Rules 3
          if ((tokens[i][1] == 'WORD' and tokens[i][2] == DISASTER and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
              (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == DISASTER and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV)):
              # print(f"Jenis Bencana: {' '.join([tokens[i][0], tokens[i+1][0]])}")
              result_jenis_bencana.append(tokens[i][0])
              result_jenis_bencana.append(tokens[i+1][0])

          # Rules 4
          if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
              (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == DISASTER and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV)):
              # print(f"Jenis Bencana: {' '.join([tokens[i][0], tokens[i+1][0]])}")
              result_jenis_bencana.append(tokens[i][0])
              result_jenis_bencana.append(tokens[i+1][0])

          # Rules 5
          if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
              (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == DISASTER and tokens[i+2][3] == LOWER_CASE and tokens[i+1][4] == OOV) and
              (tokens[i+2][1] == 'WORD' and tokens[i+2][2] == DISASTER and tokens[i+2][3] == LOWER_CASE and tokens[i+2][4] == OOV)):
              # print(f"Jenis Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
              result_jenis_bencana.append(tokens[i][0])
              result_jenis_bencana.append(tokens[i+1][0])
              result_jenis_bencana.append(tokens[i+2][0])

          # Rules 6
          if ((tokens[i][1] == 'WORD' and tokens[i][2] == DISASTER and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
              (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == DISASTER and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV)):
              # print(f"Jenis Bencana: {' '.join([tokens[i][0], tokens[i+1][0]])}")
              result_jenis_bencana.append(tokens[i][0])
              result_jenis_bencana.append(tokens[i+1][0])

          # Rules 7
          if ((tokens[i][1] == 'WORD' and tokens[i][2] == DISASTER and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV)):
              # print(f"Jenis Bencana: {' '.join([tokens[i][0]])}")
              result_jenis_bencana.append(tokens[i][0])

          # Rules 8
          if ((tokens[i][1] == 'WORD' and tokens[i][2] == DISASTER and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV)):
              # print(f"Jenis Bencana: {' '.join([tokens[i][0]])}")
              result_jenis_bencana.append(tokens[i][0])

        except IndexError:
          index_error = 1
    return set(result_jenis_bencana)



def match_waktu_bencana(tokens):
    result_waktu_bencana = []
    pengurang = 0
    if len(tokens) <= 2:
      pengurang = 0
    else:
      pengurang = 2
    for i in range(len(tokens)):
      try:
        # Rules 1
        if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TIME_FORM and tokens[i+1][4] == OOV) and
            (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == UPPER_CASE and tokens[i+2][4] == OOV)):
            # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
            result_waktu_bencana.append(tokens[i][0])
            result_waktu_bencana.append(tokens[i+1][0])
            result_waktu_bencana.append(tokens[i+2][0])

        # Rules 2
        if((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TIME_FORM and tokens[i+1][4] == OOV) and
            (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == UPPER_CASE and tokens[i+2][4] == OOV)):
            # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
            result_waktu_bencana.append(tokens[i][0])
            result_waktu_bencana.append(tokens[i+1][0])
            result_waktu_bencana.append(tokens[i+2][0])

        # Rules 3
        if((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV) and
            (tokens[i+2][1] == 'WORD' and tokens[i+2][2] == DAY and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV)):
            # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
            result_waktu_bencana.append(tokens[i][0])
            result_waktu_bencana.append(tokens[i+1][0])
            result_waktu_bencana.append(tokens[i+2][0])

        # Rules 4
        if((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV) and
            (tokens[i+2][1] == 'WORD' and tokens [i+2][2]==DAY and tokens[i+2][3] == LOWER_CASE and tokens[i+2][4] == OOV)):
            # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
            result_waktu_bencana.append(tokens[i][0])
            result_waktu_bencana.append(tokens[i+1][0])
            result_waktu_bencana.append(tokens[i+2][0])

        # Rules 5
        if((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
            (tokens[i+2][1] == 'WORD' and tokens [i+2][2]== DAY and tokens[i+2][3] == LOWER_CASE and tokens[i+2][4] == OOV)):
            # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
            result_waktu_bencana.append(tokens[i][0])
            result_waktu_bencana.append(tokens[i+1][0])
            result_waktu_bencana.append(tokens[i+2][0])

        # Rules 6
        if((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
            (tokens[i+2][1] == 'WORD' and tokens [i+2][2]== DAY and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV)):
            # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
            result_waktu_bencana.append(tokens[i][0])
            result_waktu_bencana.append(tokens[i+1][0])
            result_waktu_bencana.append(tokens[i+2][0])

        # Rules 7
        if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TIME_FORM and tokens[i+1][4] == OOV)):
            # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0]])}")
            result_waktu_bencana.append(tokens[i][0])
            result_waktu_bencana.append(tokens[i+1][0])

        # Rules 8
        if((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
            (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TIME_FORM and tokens[i+1][4] == OOV)):
            # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0]])}")
            result_waktu_bencana.append(tokens[i][0])
            result_waktu_bencana.append(tokens[i+1][0])

      except IndexError:
        index_error = 1
    return set(result_waktu_bencana)


def predict_rule_based(text):
    result = []

    # split if teks is multiple sentences
    pattern = r'(?<!\d)\. (?!\d)|(?<=\w)[!?]'
    request_teks = [k.strip() for k in re.split(pattern, text) if k.strip()]

    for teks in request_teks:
        predictions = []
        tokens = word_tokenize(teks)
        classified_tokens = [classify_token(token) for token in tokens]
        classified_contextual_features = classify_contextual_features(classified_tokens,contextual_features_dictionary)
        classified_morphological_features = classify_morphological_features(classified_contextual_features,morphological_features_dictionary)
        classified_partOfSpeech_features = classify_partOfSpeech_features(classified_morphological_features,partOfSpeech_features_dictionary)
        # Apply the rules
        lokasi = match_lokasi_bencana(classified_partOfSpeech_features)
        dampak = match_dampak_bencana(classified_partOfSpeech_features)
        jenis = match_jenis_bencana(classified_partOfSpeech_features)
        waktu = match_waktu_bencana(classified_partOfSpeech_features)

        for token in tokens:
            if token in list(lokasi):
                predictions.append({'teks':token,'label':"Lokasi"})
            elif token in list(dampak):
                predictions.append({'teks':token,'label':"Dampak"})
            elif token in list(jenis):
                predictions.append({'teks':token,'label':"Bencana"})
            elif token in list(waktu):
                predictions.append({'teks':token,'label':"Waktu"})
            else:
                predictions.append({'teks':token,'label':"Other"})
        
        result.append(predictions)
    
    return result
   



