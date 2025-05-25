from sklearn.model_selection import train_test_split
import pandas as pd

input_file = "data/complaints_university_students.csv"
train_file = "data/university_students_complaints_train.csv"
val_file = "data/university_students_complaints_val.csv"

df = pd.read_csv(input_file)

# Dividir el dataset en entrenamiento (80%) y validación (20%)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Guardar los conjuntos en archivos CSV separados
train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)

print(f"Dataset de entrenamiento guardado en {train_file}")
print(f"Dataset de validación guardado en {val_file}")
