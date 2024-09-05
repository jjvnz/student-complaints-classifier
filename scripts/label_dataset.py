import pandas as pd

# Cargar el dataset original
input_file = "university_students_complaints_es.csv"
output_file = "university_students_complaints_labeled.csv"

# Leer el archivo CSV
df = pd.read_csv(input_file)

# Verificar las primeras filas para entender la estructura
print(df.head())

# Añadir una columna de etiquetas. En este caso, todas las quejas tienen la misma etiqueta.
# Asumimos que la etiqueta '1' representa 'Queja'. Puedes ajustar esto según tus necesidades.
df['label'] = 1  # Aquí '1' es el código para 'Queja'

# Guardar el nuevo dataset con etiquetas en un nuevo archivo CSV
df.to_csv(output_file, index=False)

print(f"Dataset etiquetado guardado en {output_file}")
