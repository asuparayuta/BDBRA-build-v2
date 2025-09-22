
from io import BytesIO
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    try:
        laparams = LAParams(line_margin=0.2, word_margin=0.1, char_margin=2.0)
        return extract_text(BytesIO(pdf_bytes), laparams=laparams) or ""
    except Exception:
        return ""
