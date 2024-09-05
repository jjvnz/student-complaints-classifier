import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

# Cargar el dataset
dataset = pd.read_csv('university_students_complaints.csv')

# Cargar el modelo y el tokenizador para la traducción de inglés a español
model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Función para traducir texto
def traducir(text):
    # Tokenizar el texto
    inputs = tokenizer.encode(text, return_tensors="pt", padding=True, truncation=True)
    # Generar la traducción
    translated = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    # Decodificar el texto traducido
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Traducir las quejas (puedes cambiar 'complaints' a la columna correspondiente)
dataset['complaints_es'] = dataset['Reports'].apply(lambda x: traducir(x))

# Guardar el dataset traducido
dataset.to_csv('university_students_complaints_es.csv', index=False)

print("Traducción completada y guardada.")
