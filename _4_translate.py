from transformers import pipeline     # Transformers pipeline

model_checkpoint = "Helsinki-NLP/opus-mt-en-ru"
translator = pipeline("translation", model=model_checkpoint)
results = translator("(What does it mean, Mary?) (Why do you want to leave from me?) (I want to know.)")
print(results)  # [{'translation_text': 'у меня немного простуды и кашель'}]