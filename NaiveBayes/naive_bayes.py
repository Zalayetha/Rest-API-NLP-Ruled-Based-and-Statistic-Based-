import pprint
import pandas as pd
import re
import string
import nltk
from nltk import word_tokenize
nltk.download('punkt')
from nltk.tag import CRFTagger
import pycrfsuite
from flask_mysqldb import MySQL

# Data Training
data_training = pd.read_excel("data/data_anotation_v3.xlsx",sheet_name="5k-nostem",usecols=["currentword","currenttag",'bef1tag','token','class'])

# Remove Single Character
def removeSingleChar(text):
    return re.sub(r"\b[a-zA-Z]\b","",text)

# REMOVE PUNCTUATION
def removePunctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

# TOKENIZING
def tokenizing(text):
    return word_tokenize(text)

# to check token type
def checkTokenType(teks):
    # Regular expression for finding a word
    word_regex = r'\b[A-Za-z]+\b'
    # Regular expression for finding a number
    number_regex = r'\b\d+\b'

    is_word = bool(re.fullmatch(word_regex,teks))
    is_number = bool(re.fullmatch(number_regex,teks))

    if is_word:
        return "Word"
    elif is_number:
        return "Number"

def predict_statistic(teks):
    predicted_arr = []
    result_arr = []
    teks = removeSingleChar(teks)
    teks = removePunctuation(teks)
    teks = tokenizing(teks)


    crftagger = CRFTagger()
    crftagger.set_model_file('data/all_indo_man_tag_corpus_model.crf.tagger')
    list = crftagger.tag_sents([teks])

    teksNew = []
    for i in list:
        for j in i:
            teksNew.append(j)


    test_data = []
    for i in range(len(teksNew)):
        test_instance = {
            "currentword": teksNew[i][0],
            "currenttag": teksNew[i][1],
            "bef1tag": teksNew[i-1][1] if i > 0 else "Null",  # Handles the first word having no previous tag
            "token": checkTokenType(teksNew[i][0]),
            "class": "?"  # Assuming the class is unknown
        }
        test_data.append(test_instance)

    # Step 1: Menghitung Probabilitas Kelas
    total_data = len(data_training["class"])
    kelas_counts = {}
    for kelas in data_training["class"]:
        # if kelas != "O":
        kelas_counts[kelas] = kelas_counts.get(kelas, 0) + 1

    probabilitas_kelas = {kelas: count / total_data for kelas, count in kelas_counts.items()}

    # Menghitung jumlah total data (termasuk kelas "O" jika ada)
    total_data = sum(kelas_counts.values())

    # Step 2: Menghitung Probabilitas setiap fitur dengan Laplace Smoothing
    probabilitas_fitur = {}
    unique_values_per_feature = {fitur: set() for fitur in data_training if fitur != "class"}
    for fitur in data_training:
        if fitur != "class":
            for value in data_training[fitur]:
                unique_values_per_feature[fitur].add(value)

    for kelas in probabilitas_kelas:
        probabilitas_fitur[kelas] = {}
        for fitur in data_training:
            if fitur != "class":
                fitur_counts = {}

                for i in range(len(data_training["class"])):
                    if data_training["class"][i] == kelas:
                        fitur_counts[data_training[fitur][i]] = fitur_counts.get(data_training[fitur][i], 0) + 1

                total_fitur_in_kelas = sum(fitur_counts.values())
                V = len(unique_values_per_feature[fitur])  # Vocabulary size untuk Laplace Smoothing
                probabilitas_fitur[kelas][fitur] = {value: (count + 1) / (total_fitur_in_kelas + V)
                                                    for value, count in fitur_counts.items()}

    # Step 3: Menghitung Probabilitas Gabungan
    for dataTest in test_data:
        probabilitas_gabungan = {}
        for kelas in probabilitas_kelas:
            probabilitas_gabungan[kelas] = probabilitas_kelas[kelas]
            for fitur in dataTest:
                if fitur != "class":
                    value = dataTest[fitur]
                    if value in probabilitas_fitur[kelas][fitur]:
                        probabilitas_gabungan[kelas] *= probabilitas_fitur[kelas][fitur][value]
                    else:
                        # Laplace smoothing jika fitur dari data test tidak dikenal
                        probabilitas_gabungan[kelas] *= 1 / (kelas_counts[kelas] + len(unique_values_per_feature[fitur]))
        # Step 4: Menentukan Kelas dengan probabilitas maksimum
        predicted_class = max(probabilitas_gabungan, key=probabilitas_gabungan.get)
        predicted_arr.append(predicted_class)

    if(len(teksNew) == len(predicted_arr)):
        print(len(teksNew))
        panjang = len(teksNew)
        for i in range(panjang):
            result_arr.append({'teks':teksNew[i][0],'label':predicted_arr[i]})
    
    return result_arr
        

        

