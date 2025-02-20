import nltk
import os

custom_nltk_path = os.path.join(os.getcwd(), "nltk_data")
nltk.download('punkt', download_dir=custom_nltk_path)
nltk.download('stopwords', download_dir=custom_nltk_path)
nltk.download('punkt_tab', download_dir=custom_nltk_path)
