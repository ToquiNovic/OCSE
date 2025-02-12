from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import spacy
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type", "Authorization"],
)


# Cargar el modelo de SpaCy en español
nlp = spacy.load("es_core_news_md")

# Cargar los comandos desde el JSON
with open("opensim_commands_es.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Preparar datos para entrenamiento
descriptions = []
commands = []
categories = []

for category in data:
    for command in category["commands"]:
        descriptions.append(command["description"].lower())  
        commands.append(command["command"])
        categories.append(category["category"])

# Vectorizar las descripciones con TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(descriptions)

# Convertir la matriz TF-IDF a float32 para Faiss
X_array = X.toarray().astype('float32')

# Normalizar vectores si usamos IndexFlatIP (similaridad coseno)
X_array /= np.linalg.norm(X_array, axis=1, keepdims=True)

# Crear el índice de Faiss y agregar los datos
index = faiss.IndexFlatIP(X_array.shape[1])  # Cambié L2 por IP para similitud coseno
index.add(X_array)

def fast_search_command(query):
    query = query.lower()
    query_vec = vectorizer.transform([query]).toarray().astype('float32')

    # Normalizar el vector de consulta
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)

    # Buscar los 5 resultados más similares
    scores, indices = index.search(query_vec, 5)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if score > 0.3:  # Umbral de similitud
            results.append({
                "command": commands[idx],
                "description": descriptions[idx],
                "category": categories[idx],
                "score": round(float(score) * 100, 2)  # Convertimos la similitud en porcentaje
            })

    return results if results else [{"message": "❌ No se encontraron comandos similares."}]

# Definir el esquema para recibir el JSON
class SearchRequest(BaseModel):
    query: str

@app.post("/search")
async def search(request: SearchRequest):
    if request.query:
        results = fast_search_command(request.query)
        return JSONResponse(content=results)
    else:
        raise HTTPException(status_code=400, detail="No query provided")

# Iniciar el servidor con `python train.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
