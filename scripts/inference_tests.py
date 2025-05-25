from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("fine-tuned-model")
model = AutoModelForSequenceClassification.from_pretrained("fine-tuned-model")

def test_inferencia(texts):
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        categoria_idx = torch.argmax(outputs.logits, dim=1).item()
        
        if categoria_idx == 1:
            categoria = "Queja"
        elif categoria_idx == 2:
            categoria = "Solicitud"
        elif categoria_idx == 3:
            categoria = "Sugerencia"
        elif categoria_idx == 4:
            categoria = "Reclamo"
        else:
            categoria = "Desconocido"
            
        print(f"Texto: '{text}' -> Categoría: {categoria}")

test_inferencia([
    "Desde el 28 de January de 2025, he observado filtraciones de agua en el techo del laboratorio de Química Orgánica.",
    "Solicito revisión inmediata y protocolo de respaldo.",
    "El comedor necesita más opciones vegetarianas.",
    "Sugiero crear un comité estudiantil que colabore mensualmente con la dirección para evaluar servicios y horarios."
])
