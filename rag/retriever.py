# ─── Retriever: conector con la base de datos vectorial ChromaDB ───────────────
# Este archivo es el "puente" entre el código y ChromaDB.
# Su única responsabilidad es abrir la conexión con la BD de vectores ya construida.
#
# ¿Qué es un vector store?
#   Es una base de datos especial que no guarda texto, sino vectores numéricos.
#   Cada texto de la knowledge_base fue convertido en un vector (lista de 1536 números)
#   por el modelo de OpenAI "text-embedding-3-small".
#   Cuando hacemos una búsqueda, también convertimos la pregunta en vector
#   y buscamos los vectores más "cercanos" (similares en significado).

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_vectorstore() -> Chroma:
    """
    Abre la conexión con ChromaDB y devuelve un objeto listo para hacer búsquedas.
    No crea nada nuevo — la BD debe existir previamente (ejecutar build_vector_db.py).
    """

    # Calculamos la ruta absoluta a data/chroma_db/ sea cual sea el directorio de trabajo
    # __file__ es la ruta de este archivo → .parent es rag/ → .parent es la raíz del proyecto
    persist_dir = str(Path(__file__).parent.parent / "data" / "chroma_db")

    # El modelo de embeddings DEBE ser el mismo que se usó al crear la BD
    # Si usásemos un modelo diferente, los vectores no serían comparables y las búsquedas fallarían
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Abrimos la colección existente en disco
    # collection_name="knowledge_base" → nombre que le dimos al crear la BD
    # persist_directory → carpeta donde ChromaDB guardó los archivos binarios
    return Chroma(
        collection_name="knowledge_base",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
