from transformers import pipeline     # Transformers pipeline

model_checkpoint = "Helsinki-NLP/opus-mt-en-ru"
translator = pipeline("translation", model=model_checkpoint)

def translate_text(text):
    """
    Translate the input text to Russian using the specified model.
    
    Args:
        text (str): The text to translate.
        
    Returns:
        list: A list of dictionaries containing the translation.
    """
    transletion = translator(text)
    return transletion[0]['translation_text']

if __name__ == '__main__':
    results = translator("(What does it mean, Mary?) (Why do you want to leave from me?) (I want to know.)")
    print(results)  # [{'translation_text': 'у меня немного простуды и кашель'}]