from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the pre-trained T5 model and tokenizer
model_name = "t5-small"  # You can also use 't5-base' or 't5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# French sentence to translate
french_sentence = "Trudeau et Colbert se lient autour du statut partagé de gars qui étaient cool il y a dix ans."

# Prepare the input with the appropriate task prefix
input_text = f"translate French to English: {french_sentence}"
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate the translation
output = model.generate(
    inputs['input_ids'],
    max_length=150,
    num_beams=4,
    early_stopping=True
)

# Decode the generated tokens
translated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)

print("Translated Sentence:", translated_sentence)
