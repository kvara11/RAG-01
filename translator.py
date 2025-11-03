
from deep_translator import GoogleTranslator

def translate(text: str, to_lang: str = "en") -> str:
    result = GoogleTranslator(source='auto', target=to_lang).translate(text)
    return result
