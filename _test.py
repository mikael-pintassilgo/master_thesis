
""" 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/translation-en-pt-t5")
model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-en-pt-t5")

# Create the translation pipeline
translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)

# Example translation
result = translation_pipeline("I like to eat rice.", src_lang="en", tgt_lang="pt")
print(result)
"""

from transformers import M2M100ForConditionalGeneration, pipeline
from transformers import M2M100ForConditionalGeneration
from tokenization_small100 import SMALL100Tokenizer

# Load the tokenizer and model
model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")

# Create the translation pipeline
translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)



tokenizer.tgt_lang = "pt"
encoded_en = tokenizer("I like to eat rice.", return_tensors="pt")
generated_tokens = model.generate(**encoded_en)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)