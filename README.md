# StudentComplaintsClassifier

## Descripción

Este proyecto emplea el modelo **DistilBERT** para la clasificación de quejas de estudiantes sobre la infraestructura universitaria. El objetivo es entrenar un modelo de aprendizaje automático capaz de identificar y clasificar quejas en diferentes categorías basadas en el contenido textual de las quejas.

## Estructura del Proyecto


```bash
StudentComplaintsClassifier/
│
├── data/
│   ├── university_students_complaints_es.csv
│   ├── university_students_complaints_labeled.csv
│   ├── university_students_complaints_train.csv
│   └── university_students_complaints_val.csv
│
├── scripts/
│   ├── generate_data.py
│   ├── separate_dataset.py
│   ├── run_tokenizer.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── test_inference.py
│   └── gradio_interface.py
│
├── results/
│   └── evaluation_report.txt
│
├── train_dataset/
│
├── val_dataset/
│
└── fine-tuned-model/
    ├── config.json
    ├── model.safetensors
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt

```


- **`scripts/`**: Contiene los scripts de Python para el procesamiento y entrenamiento.
  - **`label_data.py`**: Script para etiquetar los datos.
  - **`separate_dataset.py`**: Script para dividir los datos en conjuntos de entrenamiento y validación.
  - **`run_tokenizer.py`**: Script para tokenizar los datos utilizando el tokenizer de DistilBERT.
  - **`train_model.py`**: Script para entrenar el modelo DistilBERT con los datos tokenizados.
- **`data/`**: Contiene los archivos CSV con quejas.
  - **`university_students_complaints_es.csv`**: Archivo CSV con las quejas originales en español.
  - **`university_students_complaints_labeled.csv`**: Archivo CSV con las quejas etiquetadas.
  - **`university_students_complaints_train.csv`**: Archivo CSV con los datos de entrenamiento.
  - **`university_students_complaints_val.csv`**: Archivo CSV con los datos de validación.
- **`train_dataset/`**: Datos tokenizados para entrenamiento.
- **`val_dataset/`**: Datos tokenizados para validación.
- **`fine-tuned-model/`**: Contiene el modelo entrenado.
- **`results/`**: Contiene los resultados de la evaluación del modelo.
- **`README.md`**: Este archivo.
- **`requirements.txt`**: Archivo con los requisitos del proyecto.

## Requisitos

- **Python 3.10** o superior
- **transformers**
- **datasets**
- **pandas**
- **scikit-learn**
- **gradio**

Instala los requisitos utilizando `pip`:

```bash
pip install -r requirements.txt
```

## Instrucciones para Ejecutar el Proyecto

### 1. Preparar los Datos

1. Genera datos datos ejecutando el script `scripts/generate_data.py`.
2. Divide los datos en conjuntos de entrenamiento y validación, ejecutando el script `scripts/separate_dataset.py`.

### 2. Tokenizar los Datos

Ejecuta el script `scripts/run_tokenizer.py` para tokenizar los datos:

```bash
python scripts/run_tokenizer.py
```

### 3. Entrenar el Modelo

Ejecuta el script `scripts/train_model.py` para entrenar el modelo:

```bash
python scripts/train_model.py
```

### 4. Guardar el Modelo Entrenado

El modelo entrenado se guardará en el directorio `./fine-tuned-model`.

### 5. Correr modelo con Gradio

```bash
python scripts/gradio_interface.py
```

Abre la url http://localhost:7860 en el navegador.


## Resultados

El modelo será evaluado durante el proceso de entrenamiento, y los resultados se guardarán en el directorio `./results`.


## Evaluación del Modelo

Después de entrenar el modelo, se realizó una evaluación utilizando el conjunto de validación. Los resultados fueron:

- **Precisión**: 1.00
- **Recall**: 1.00
- **F1-Score**: 1.00
- **Exactitud (Accuracy)**: 1.00

Estos resultados indican un rendimiento perfecto en el conjunto de validación. 

## Contribuciones

Las contribuciones son bienvenidas. Por favor, sigue estos pasos para contribuir:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -am 'Añadir nueva funcionalidad'`).
4. Empuja los cambios a tu fork (`git push origin feature/nueva-funcionalidad`).
5. Crea un Pull Request.

## Licencia

Este proyecto está licenciado bajo la **Licencia MIT**. Consulta el archivo `LICENSE` para más detalles.

## Contacto

Si tienes preguntas o comentarios, puedes contactarme a través de [jjvnz.dev@outlook.com](mailto:jjvnz.dev@outlook.com).