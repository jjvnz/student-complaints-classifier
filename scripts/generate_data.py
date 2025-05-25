import os
import random
import logging
import pandas as pd
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv
import os
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pqrs(BaseModel):
    pqrs: str
    label: int


# Set up the Google GenAI client with the API key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=API_KEY)

pqrs_data = []
response = client.models.generate_content(
    model="gemini-2.0-flash", 
    contents="Genera una lista de 50 PQRS (Peticiones, Quejas, Reclamos y Sugerencias) expresadas por estudiantes universitarios. Cada entrada debe estar redactada en primera persona y reflejar el lenguaje natural, coloquial y auténtico de un estudiante. Es indispensable que cada PQRS contenga al menos 15 palabras para asegurar suficientes detalles y contexto. Además, cada PQRS debe incluir un campo 'label' según el tipo, asignándole el valor correspondiente: > - Peticiones: 2 > - Quejas: 1 > - Reclamos: 4 > - Sugerencias: 3 > Las entradas deben abordar una amplia variedad de temas relacionados con la experiencia universitaria, como la calidad de la enseñanza, la organización de clases, la infraestructura del campus, la atención administrativa, los servicios estudiantiles, actividades extracurriculares, entre otros. Asegúrate de que cada PQRS sea única y refleje problemas, inquietudes o propuestas reales y específicas.",
    config={
        "response_mime_type": "application/json",
        "response_schema": list[Pqrs],
    },
)

parsed_response: list[Pqrs] = response.parsed
pqrs_data = pqrs_data + list(map(lambda pqrs: { "complaints": pqrs.pqrs, "label": pqrs.label }, parsed_response))
file_path = "data/complaints_university_students.csv"

random.shuffle(pqrs_data)

if os.path.exists(file_path):
    data = pd.DataFrame(pqrs_data)
    df = pd.read_csv(file_path)
    df = pd.concat([df, data], ignore_index=True)
else:
    df = pd.DataFrame(pqrs_data)

df.to_csv(file_path, index=False)
