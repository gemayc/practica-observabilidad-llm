# ─── Agente RAG: busca en la base de conocimiento antes de responder ──────────
# Este agente se activa cuando la pregunta es sobre los temas de nuestra BD:
# LangSmith, LangChain, LangGraph, RAG, GraphRAG, ChromaDB, métricas, trazas...
#
# Flujo de este agente:
#   1. Llama a GraphRAG.retrieve() → busca en ChromaDB + expande por el grafo
#   2. Recibe el contexto enriquecido (textos relevantes + relaciones del grafo)
#   3. Construye un prompt que incluye ese contexto + la pregunta del usuario
#   4. Llama a ChatGPT con ese prompt para que responda basándose en los datos

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

# Importamos los prompts del agente RAG
from prompts.rag_prompt import RAG_HUMAN, RAG_SYSTEM

# Importamos la clase GraphRAG que hace la búsqueda vectorial + expansión por grafo
from rag.graph_rag import GraphRAG

# Variable global para almacenar la instancia de GraphRAG
# Patrón Singleton: creamos GraphRAG UNA SOLA VEZ y la reutilizamos en todas las llamadas
# Esto evita recargar ChromaDB y el grafo JSON en cada petición (es costoso en tiempo)
_graph_rag: GraphRAG | None = None


def _get_graph_rag() -> GraphRAG:
    """Devuelve la instancia única de GraphRAG, creándola si no existe todavía."""
    global _graph_rag

    # Si todavía no se ha creado, la creamos ahora
    if _graph_rag is None:
        _graph_rag = GraphRAG()

    # Devolvemos siempre la misma instancia ya cargada
    return _graph_rag


# run_type="retriever" indica a LangSmith que este span hace recuperación de información
@traceable(name="rag_agent", run_type="retriever")
def run_rag_agent(query: str) -> str:
    """Busca contexto en la BD de conocimiento y responde basándose en él."""

    # Paso 1: recuperamos el contexto enriquecido (GraphRAG = vectores + grafo)
    # k=3 → traer los 3 documentos más similares como punto de partida
    graph_rag = _get_graph_rag()
    context = graph_rag.retrieve(query, k=3)

    # Paso 2: inicializamos el modelo con temperatura baja para respuestas fieles al contexto
    # temperature=0.2 → casi determinista, no queremos que el modelo "se invente" cosas
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # Paso 3: construimos el prompt inyectando el contexto recuperado dentro de RAG_HUMAN
    # El modelo ve: "Contexto: [textos de ChromaDB + relaciones del grafo]\nPregunta: [query]"
    messages = [
        SystemMessage(content=RAG_SYSTEM),
        HumanMessage(content=RAG_HUMAN.format(context=context, query=query)),
    ]

    # Paso 4: llamamos a la API y devolvemos la respuesta basada en el contexto
    response = llm.invoke(messages)
    return response.content
