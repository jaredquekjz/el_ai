import numpy as np
import pandas as pd
from nomic import atlas

# Load your CSV file
csv_path = "/Users/e/Python/ChatGPT Programs/EL AI (Storage)/Embeddings/combined_embeddings.csv"
df = pd.read_csv(csv_path, names=['combined', 'embedding'], header=0)

# Add an 'ID' column to the DataFrame
df['ID'] = df.index

print("Column names:", df.columns)
print("First few rows of the DataFrame:")
print(df.head())

# Convert the 'embedding' field to a NumPy array
embeddings = np.array(df['embedding'].apply(
    lambda x: [float(val) for val in x.strip('[]').split(',')]).tolist())

# Create a list of dictionaries for the 'data' parameter
data = df[['ID', 'combined']].to_dict(orient='records')

# Create the map
project = atlas.map_embeddings(
    embeddings=embeddings,
    data=data,
    id_field='ID',  # Update the field name to match the ID column in your CSV file
    colorable_fields=['combined'],
    name='El's Brain',
    description='Embeddings of EL'
)
