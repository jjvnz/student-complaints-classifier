from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cargar el modelo y el tokenizer entrenados
model = AutoModelForSequenceClassification.from_pretrained("./fine-tuned-model")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")

# Guardar el modelo y el tokenizer
model.save_pretrained("fine-tuned-model")
tokenizer.save_pretrained("fine-tuned-model")

print("Modelo y tokenizer guardados en 'fine-tuned-model'.")
