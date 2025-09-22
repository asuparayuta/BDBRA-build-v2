
import re
from html import unescape
def html_to_text(html: str) -> str:
    html = re.sub(r'<(script|style)[\s\S]*?</\1>', ' ', html, flags=re.I)
    html = re.sub(r'</?(br|p|div|li|tr|h\d)>', '\n', html, flags=re.I)
    text = re.sub(r'<[^>]+>', ' ', html)
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
