# ─── GraphRAG: recuperación de información con grafo de conocimiento ──────────
# Este es el corazón del sistema RAG. Combina DOS técnicas de búsqueda:
#
#   1. Búsqueda vectorial (ChromaDB):
#      Convierte la pregunta en vector y busca los K textos más similares.
#      Es como buscar por "significado" en lugar de por palabras exactas.
#
#   2. Expansión por grafo (NetworkX):
#      Los nodos encontrados en el paso 1 pueden tener "vecinos" en el grafo.
#      Por ejemplo: si encontramos "RAG", el grafo nos dice que se relaciona con
#      "ChromaDB" y "GraphRAG". Los añadimos al contexto aunque no aparecieran
#      en la búsqueda vectorial.
#
# El resultado es un contexto MÁS RICO que el RAG tradicional.

import json
from pathlib import Path

import networkx as nx

from rag.retriever import get_vectorstore


class GraphRAG:
    """Combina búsqueda vectorial (ChromaDB) con expansión por grafo (NetworkX)."""

    def __init__(self):
        # Usamos None como valor inicial → carga perezosa (lazy loading)
        # Los datos pesados se cargan sólo cuando se necesitan por primera vez
        self._graph: nx.DiGraph | None = None
        self._vectorstore = None

    @property
    def graph(self) -> nx.DiGraph:
        """Carga el grafo de conocimiento desde JSON la primera vez que se accede."""
        if self._graph is None:
            # Calculamos la ruta al archivo JSON del grafo
            graph_path = Path(__file__).parent.parent / "data" / "knowledge_graph.json"

            if not graph_path.exists():
                raise FileNotFoundError(
                    "Knowledge graph not found. Run: python scripts/build_knowledge_graph.py"
                )

            # Leemos el JSON y lo reconstruimos como grafo de NetworkX
            with open(graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # node_link_graph() convierte el formato JSON de vuelta a un DiGraph
            self._graph = nx.node_link_graph(data)

        return self._graph

    @property
    def vectorstore(self):
        """Abre la conexión con ChromaDB la primera vez que se accede."""
        if self._vectorstore is None:
            chroma_path = Path(__file__).parent.parent / "data" / "chroma_db"

            if not chroma_path.exists():
                raise FileNotFoundError(
                    "Vector DB not found. Run: python scripts/build_vector_db.py"
                )

            # Reutilizamos la función del retriever para abrir ChromaDB
            self._vectorstore = get_vectorstore()

        return self._vectorstore

    def retrieve(self, query: str, k: int = 3) -> str:
        """
        Recupera contexto enriquecido para la pregunta dada.
        Combina búsqueda vectorial + expansión 1 salto en el grafo.
        Devuelve un string de texto listo para insertar en el prompt del LLM.
        """

        # ── Paso 1: Búsqueda vectorial ─────────────────────────────────────────
        # Buscamos los K documentos más similares a la pregunta en ChromaDB
        # "similares" significa que sus vectores están más cerca en el espacio matemático
        results = self.vectorstore.similarity_search(query, k=k)

        # Guardamos los IDs de los nodos encontrados directamente por similitud
        direct_ids = [doc.metadata["id"] for doc in results]

        # ── Paso 2: Expansión por el grafo (1 salto) ───────────────────────────
        # Para cada nodo encontrado, buscamos sus vecinos en el grafo:
        # - successors  → nodos a los que apunta (relaciones salientes: "RAG → usa → ChromaDB")
        # - predecessors → nodos que apuntan a él (relaciones entrantes: "GraphRAG → extiende → RAG")
        expanded_ids: set[str] = set(direct_ids)
        for node_id in direct_ids:
            if node_id in self.graph:
                expanded_ids.update(self.graph.successors(node_id))
                expanded_ids.update(self.graph.predecessors(node_id))

        # ── Paso 3: Construimos el texto de contexto ───────────────────────────
        context_parts: list[str] = []

        # Primero añadimos los resultados directos de la búsqueda vectorial
        for doc in results:
            title = doc.metadata["title"]
            context_parts.append(f"### {title}\n{doc.page_content}")

        # Luego añadimos los nodos expandidos por el grafo que NO estaban en la búsqueda directa
        for node_id in expanded_ids - set(direct_ids):
            node_data = self.graph.nodes.get(node_id, {})
            content = node_data.get("content", "")
            title = node_data.get("title", node_id)

            # Solo añadimos si el nodo tiene contenido (puede haber nodos vacíos)
            if content:
                context_parts.append(f"### {title} (contexto relacionado)\n{content}")

        # ── Paso 4: Añadimos las relaciones del grafo ──────────────────────────
        # Esto le dice al LLM cómo se relacionan los conceptos entre sí
        # Ejemplo: "LangSmith → provides → Distributed Tracing"
        relation_lines: list[str] = []
        for src, dst, data in self.graph.edges(data=True):
            # Solo incluimos aristas donde AMBOS extremos están en nuestro conjunto expandido
            if src in expanded_ids and dst in expanded_ids:
                src_title = self.graph.nodes[src].get("title", src)
                dst_title = self.graph.nodes[dst].get("title", dst)
                relation_lines.append(
                    f"- **{src_title}** → _{data['relationship']}_ → **{dst_title}**"
                )

        # Añadimos las relaciones al contexto (máximo 12 para no saturar el prompt)
        if relation_lines:
            context_parts.append("### Relaciones entre conceptos\n" + "\n".join(relation_lines[:12]))

        # Unimos todas las partes con doble salto de línea para que el LLM las lea bien
        return "\n\n".join(context_parts)
