from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from sklearn.metrics import classification_report

# Cargar el modelo y el tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./fine-tuned-model")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")

# Cargar los datos de validación
val_data = pd.read_csv('data/university_students_complaints_val.csv')

# Extraer las quejas y las etiquetas verdaderas
texts = val_data['complaints'].tolist()
true_label = val_data['label'].tolist()  # Asegúrate de que 'label' es el nombre correcto de la columna

# Tokenizar los datos
inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

# Realizar inferencia
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).tolist()

# Calcular y mostrar métricas de evaluación
report = classification_report(true_label, predictions, target_names=["Queja"])
print(report)

# Guardar el informe de clasificación en un archivo
with open('results/evaluation_report.txt', 'w') as f:
    f.write(report)
