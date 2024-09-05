# scripts/inferencia_tests.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cargar el modelo y el tokenizer guardados
tokenizer = AutoTokenizer.from_pretrained("fine-tuned-model")
model = AutoModelForSequenceClassification.from_pretrained("fine-tuned-model")

def test_inferencia(texts):
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        categoria_idx = torch.argmax(outputs.logits, dim=1).item()
        
        # Asume que la única categoría es "Queja"
        if categoria_idx == 1:  # 1 es el índice que devuelve "Queja"
            categoria = "Queja"
        else:
            categoria = "Desconocido"
            
        print(f"Texto: '{text}' -> Categoría: {categoria}")

# Prueba con algunos ejemplos
test_inferencia([
    "Necesito un aumento en la calidad del wifi en la biblioteca.",
    "El comedor necesita más opciones vegetarianas.",
])
