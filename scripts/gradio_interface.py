import torch
import gradio as gr
from google import genai
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
)
import os

tokenizer = AutoTokenizer.from_pretrained("fine-tuned-model")
model = AutoModelForSequenceClassification.from_pretrained("fine-tuned-model")
# Set up the Google GenAI client with the API key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=API_KEY)

def map_category(category_idx: int):
    if category_idx == 1:
        return "Queja"
    elif category_idx == 2:
        return "Solicitud"
    elif category_idx == 3:
        return "Sugerencia"
    elif category_idx == 4:
        return "Reclamo"
    else:
        return "Desconocido"

def manage_pqrs(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    category = map_category(torch.argmax(outputs.logits, dim=1).item())

    prompt = (
        f"Responde de manera concisa, cordial y empática a la siguiente pqrs: \"{text}\", "
        f"la respuesta debe iniciar con la frase, \"Su {category} ha sido procesada correctamente\", "
        f"tu respuesta no debe contener caracteres especiales como corchetes u elementos modificables."
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=prompt
    )

    return response.text

iface = gr.Interface(
    fn=manage_pqrs,
    inputs=gr.Textbox(lines=5, placeholder="Escribe aquí el texto a clasificar...", label="Texto de entrada"),
    outputs=gr.Textbox(label="Resultado"),
    title="Clasificador de PQRS",
    description="Ingresa un texto y el modelo te devolverá una clasificación (por ejemplo, Queja, Sugerencia, Solicitud)."
)

if __name__ == "__main__":
    iface.launch()
