import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Remove noise, stopwords and stem the text."""
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower().split()
    text = [ps.stem(w) for w in text if w not in stop_words]
    return ' '.join(text)