import pdfplumber
import pandas as pd
import re

def detect_file_type(file):
    if file.endswith("pdf"):
        return "PDF"
    elif file.endswith("csv"):
        return "CSV"
    else:
        return "Text"
    
def parse_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text += page.extract_text() + "\n"
    return text

def parse_csv(file):
    df = pd.read_csv(file)
    return df

def parse_txt(file):
    return file.read().decode("utf-8")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text