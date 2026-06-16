# ─── Script: construye la base de datos vectorial con ChromaDB ────────────────
# Este script se ejecuta UNA SOLA VEZ para preparar los datos.
# Lee los textos de knowledge_base.py, los convierte en vectores numéricos
# usando la API de OpenAI, y los guarda en disco en data/chroma_db/.
#
# ¿Qué es un vector (embedding)?
#   Es una lista de 1536 números que representa el SIGNIFICADO de un texto.
#   Textos con significados parecidos tendrán vectores parecidos (cercanos en el espacio).
#   Esto nos permite buscar por "similitud semántica" en lugar de palabras exactas.
#
# Ejecución: uv run python scripts/build_vector_db.py
# Resultado: crea la carpeta data/chroma_db/ con archivos binarios de ChromaDB

import sys
from pathlib import Path

# Añadimos la raíz del proyecto al path para importar módulos locales
sys.path.insert(0, str(Path(__file__).parent.parent))

# Cargamos las variables de entorno (.env) antes que nada
# Necesitamos OPENAI_API_KEY para llamar a la API de embeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Importamos los textos de la base de conocimiento
from data.knowledge_base import NODES


def build_vector_db() -> Chroma:
    """
    Convierte los textos de knowledge_base.py en vectores y los guarda en ChromaDB.
    Devuelve el objeto vectorstore listo para hacer búsquedas.
    """

    # Carpeta donde ChromaDB guardará los archivos binarios en disco
    persist_dir = str(Path(__file__).parent.parent / "data" / "chroma_db")

    # Modelo de OpenAI que convierte texto → vector de 1536 dimensiones
    # Es importante usar SIEMPRE el mismo modelo en construcción y en búsqueda
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Convertimos cada nodo de la knowledge_base en un Document de LangChain
    # - page_content: el texto que se vectorizará y se buscará
    # - metadata: datos extra que se guardan junto al vector para poder recuperarlos
    docs = [
        Document(
            page_content=node["content"],
            metadata={
                "id": node["id"],       # necesario para cruzar con el grafo en GraphRAG
                "type": node["type"],
                "title": node["title"],
            },
        )
        for node in NODES
    ]

    # Extraemos los IDs de cada documento para poder actualizarlos o borrarlos después
    ids = [node["id"] for node in NODES]

    # Creamos la colección en ChromaDB y guardamos todos los documentos con sus vectores
    # from_documents() hace TODO en un solo paso:
    #   1. Llama a OpenAI para vectorizar cada texto (una llamada a la API por documento)
    #   2. Guarda los vectores + metadata en disco en persist_directory
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="knowledge_base",   # nombre de la colección (debe coincidir en retriever.py)
        persist_directory=persist_dir,
        ids=ids,
    )

    # Confirmamos que todo fue bien
    print(f"[OK] Base de datos vectorial creada en: {persist_dir}")
    print(f"     Documentos indexados: {len(docs)}")
    return vectorstore


# Punto de entrada: solo se ejecuta si llamamos directamente a este archivo
if __name__ == "__main__":
    build_vector_db()
