from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar el dataset etiquetado
input_file = "university_students_complaints_labeled.csv"
train_file = "university_students_complaints_train.csv"
val_file = "university_students_complaints_val.csv"

# Leer el archivo CSV
df = pd.read_csv(input_file)

# Dividir el dataset en entrenamiento y validación
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)  # 80% entrenamiento, 20% validación

# Guardar los conjuntos en archivos CSV separados
train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)

print(f"Dataset de entrenamiento guardado en {train_file}")
print(f"Dataset de validación guardado en {val_file}")
