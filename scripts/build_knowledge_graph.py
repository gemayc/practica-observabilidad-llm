# ─── Script: construye el grafo de conocimiento y lo guarda como JSON ─────────
# Este script se ejecuta UNA SOLA VEZ para preparar los datos.
# Lee los nodos y aristas definidos en knowledge_base.py,
# construye un grafo dirigido con NetworkX y lo guarda en data/knowledge_graph.json.
#
# Ejecución: uv run python scripts/build_knowledge_graph.py
# Resultado: crea el archivo data/knowledge_graph.json

import json
import sys
from pathlib import Path

# Añadimos la raíz del proyecto al path de Python para poder importar módulos locales
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

# Importamos los datos: NODES (lista de nodos) y EDGES (lista de aristas)
from data.knowledge_base import EDGES, NODES


def build_graph() -> nx.DiGraph:
    """
    Construye el grafo de conocimiento desde los datos de knowledge_base.py.
    Guarda el resultado en data/knowledge_graph.json y devuelve el grafo.
    """

    # Creamos un grafo DIRIGIDO (DiGraph): las aristas tienen dirección
    # Ejemplo: "RAG → usa → ChromaDB" (no es lo mismo al revés)
    G = nx.DiGraph()

    # Añadimos cada nodo al grafo con todos sus atributos (id, type, title, content)
    # **node desempaqueta el diccionario como kwargs: add_node("rag", type="pattern", title="RAG", ...)
    for node in NODES:
        G.add_node(node["id"], **node)

    # Añadimos cada arista con su etiqueta de relación
    # Formato de EDGES: (nodo_origen, nodo_destino, nombre_relacion)
    # Ejemplo: ("rag", "vector_db", "uses")
    for src, dst, relationship in EDGES:
        G.add_edge(src, dst, relationship=relationship)

    # Calculamos la ruta de salida y creamos la carpeta data/ si no existe
    output_path = Path(__file__).parent.parent / "data" / "knowledge_graph.json"
    output_path.parent.mkdir(exist_ok=True)

    # Convertimos el grafo a formato JSON usando node_link_data()
    # Este formato guarda nodos y aristas de forma que nx.node_link_graph() puede reconstruirlo
    data = nx.node_link_data(G)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Mostramos confirmación con el número de nodos y aristas creados
    print(f"[OK] Knowledge graph guardado en: {output_path}")
    print(f"     Nodos: {G.number_of_nodes()}  |  Aristas: {G.number_of_edges()}")
    return G


# Punto de entrada: solo se ejecuta si llamamos directamente a este archivo
if __name__ == "__main__":
    build_graph()
