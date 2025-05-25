from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from transformers import DistilBertTokenizerFast
import pandas as pd

# Cargar el tokenizer y el modelo
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

# Cargar los datasets tokenizados
train_dataset = load_from_disk('train_dataset')
val_dataset = load_from_disk('val_dataset')

# Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Crear el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')

print("Modelo entrenado y guardado.")
