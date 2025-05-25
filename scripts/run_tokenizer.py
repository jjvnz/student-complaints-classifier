from transformers import DistilBertTokenizerFast
from datasets import Dataset
import pandas as pd

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Cargar los datasets
train_df = pd.read_csv("data/university_students_complaints_train.csv")
val_df = pd.read_csv("data/university_students_complaints_val.csv")

# Convertir a objetos Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def tokenize_function(examples):
    return tokenizer(examples['complaints'], padding="max_length", truncation=True)

# Aplicar la tokenización
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Guardar los datasets tokenizados si es necesario
train_dataset.save_to_disk('train_dataset')
val_dataset.save_to_disk('val_dataset')

print("Datasets tokenizados guardados.")
