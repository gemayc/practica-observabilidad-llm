"""
Streamlit app — Multi-Agent LLM Observability Assistant
Ejecutar con: uv run streamlit run app.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Configuración de página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Observability Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>

    /* ── Tamaño base de TODA la app ─────────────────────────────────────────── */
    /* Aumentamos el font-size raíz: todo lo que usa "em" o "rem" se escala solo */
    html, body, [class*="css"] {
        font-size: 18px !important;
    }

    /* ── Texto general del cuerpo principal ─────────────────────────────────── */
    p, li, span, label, div {
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }

    /* ── Títulos (h1 h2 h3 h4) ──────────────────────────────────────────────── */
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.9rem !important; }
    h3 { font-size: 1.6rem !important; }
    h4, h5 { font-size: 1.35rem !important; }

    /* ── Pestañas (tabs) ─────────────────────────────────────────────────────── */
    /* El selector apunta a los botones internos de las pestañas de Streamlit */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
    }

    /* ── Botones ─────────────────────────────────────────────────────────────── */
    .stButton > button {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
        border-radius: 10px !important;
        min-height: 52px !important;
    }

    /* ── Caja de chat input (donde se escribe la pregunta) ───────────────────── */
    .stChatInput textarea {
        font-size: 1.1rem !important;
        padding: 14px !important;
    }

    /* ── Mensajes del chat (burbujas de usuario y asistente) ─────────────────── */
    .stChatMessage p,
    .stChatMessage li,
    .stChatMessage span {
        font-size: 1.05rem !important;
        line-height: 1.75 !important;
    }

    /* ── Sidebar: todo el texto ──────────────────────────────────────────────── */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        font-size: 1.0rem !important;
    }
    [data-testid="stSidebar"] h2 { font-size: 1.5rem !important; }
    [data-testid="stSidebar"] h3 { font-size: 1.25rem !important; }

    /* ── Tablas en markdown ──────────────────────────────────────────────────── */
    table, th, td {
        font-size: 1.0rem !important;
    }

    /* ── Expanders (las secciones desplegables) ──────────────────────────────── */
    .streamlit-expanderHeader {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 12px 0 !important;
    }

    /* ── Texto de info/warning/success (cajas de aviso) ─────────────────────── */
    .stAlert p { font-size: 1.0rem !important; }

    /* ── Código inline y bloques de código ───────────────────────────────────── */
    code, pre { font-size: 0.98rem !important; }

    /* ── Header principal de la app ─────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 28px 36px;
        border-radius: 14px;
        margin-bottom: 24px;
    }

    /* ── Badges de agente (colores por tipo) ─────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 20px;
        font-size: 1.0rem !important;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .badge-rag        { background: #5b21b6; color: #ede9fe; }
    .badge-code       { background: #14532d; color: #d1fae5; }
    .badge-researcher { background: #1e3a5f; color: #dbeafe; }

    /* ── Caja de respuesta con borde izquierdo de color ─────────────────────── */
    .response-box {
        border-left: 4px solid #4f46e5;
        padding: 16px 20px;
        background: #1e1e2e;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Recursos cacheados ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando agentes y modelos...")
def load_workflow():
    from graph.workflow import build_workflow
    return build_workflow()


@st.cache_data
def get_nodes():
    from data.knowledge_base import NODES
    return NODES


@st.cache_resource
def get_graph_figure():
    graph_path = Path(__file__).parent / "data" / "knowledge_graph.json"
    with open(graph_path, encoding="utf-8") as f:
        data = json.load(f)

    import networkx as nx

    G = nx.node_link_graph(data)

    type_colors = {
        "concept":   "#4ECDC4",
        "tool":      "#FF6B6B",
        "framework": "#45B7D1",
        "pattern":   "#FFA07A",
        "provider":  "#98D8C8",
    }
    node_colors = [type_colors.get(G.nodes[n].get("type", "concept"), "#aaa") for n in G.nodes]
    labels = {n: G.nodes[n].get("title", n) for n in G.nodes}

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    pos = nx.spring_layout(G, seed=42, k=2.8)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#555", arrows=True, arrowsize=14,
        width=1.3, alpha=0.65,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=2000, alpha=0.92,
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=7.5, font_color="white", font_weight="bold",
    )

    edge_labels = {(u, v): d["relationship"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=6, font_color="#aaa", bbox={"alpha": 0},
    )

    patches = [
        mpatches.Patch(color=c, label=t.capitalize())
        for t, c in type_colors.items()
    ]
    legend = ax.legend(
        handles=patches, loc="upper left",
        framealpha=0.4, labelcolor="white", facecolor="#1E1E1E",
        fontsize=9,
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 Multi-Agent Assistant")
    st.caption("Observabilidad LLM · LangSmith + LangGraph")
    st.divider()

    st.markdown("### ¿Qué puede hacer?")
    st.markdown(
        """
        | Agente | Tipo de pregunta |
        |--------|-----------------|
        | 🔬 **Researcher** | Conceptos, explicaciones, comparaciones |
        | 💻 **Code** | Generar o explicar código Python |
        | 🕸️ **GraphRAG** | Consultas sobre la base de conocimiento |
        """
    )
    st.divider()

    st.markdown("### 📖 Tema de la BD")
    st.info(
        "La base de conocimiento cubre **Observabilidad en LLMs**:\n\n"
        "LangSmith · LangChain · LangGraph · RAG · GraphRAG · "
        "ChromaDB · Métricas · Trazas · Evaluación · Agentes · "
        "Prompt Engineering · OpenAI"
    )
    st.divider()

    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "practica-observabilidad-llm")

    if tracing and api_key and api_key != "your_langsmith_api_key_here":
        st.success(f"✅ **LangSmith activo**\nProyecto: `{project}`")
    else:
        st.warning(
            "⚠️ **LangSmith inactivo**\n"
            "Edita `.env` y añade tu `LANGCHAIN_API_KEY`\n"
            "→ [smith.langchain.com](https://smith.langchain.com)"
        )

# ─── Header principal ──────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="main-header">
        <h1 style="color:white; margin:0">🤖 LLM Observability Assistant</h1>
        <p style="color:#94a3b8; margin:4px 0 0 0">
            Sistema multi-agente · LangGraph · GraphRAG con NetworkX + ChromaDB · LangSmith
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Pestañas ─────────────────────────────────────────────────────────────────

tab_chat, tab_kb, tab_graph, tab_explain = st.tabs(
    ["💬 Chat", "📚 Base de Conocimiento", "🕸️ Grafo", "❓ Cómo funciona"]
)

# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑA 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    # Inicializar historial
    if "history" not in st.session_state:
        st.session_state.history = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # ── Sugerencias ────────────────────────────────────────────────────────────
    st.markdown("##### 💡 Preguntas de ejemplo — pulsa una para enviarla:")
    suggestions = [
        ("🕸️ ¿Qué es LangSmith?",          "¿Qué es LangSmith y para qué sirve?"),
        ("🕸️ RAG vs GraphRAG",              "¿Qué diferencia hay entre RAG y GraphRAG?"),
        ("🕸️ Métricas en producción",       "¿Qué métricas clave debo monitorizar en un sistema LLM en producción?"),
        ("💻 Código: traza con LangSmith",   "Escribe un ejemplo en Python que use LangChain con trazas en LangSmith"),
        ("💻 Agente con LangGraph",          "Escribe un agente simple con LangGraph en Python"),
        ("🔬 ¿Por qué observabilidad?",      "¿Por qué es importante la observabilidad en sistemas LLM?"),
    ]

    cols = st.columns(3)
    for i, (label, prompt) in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(label, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_query = prompt
                st.rerun()

    st.divider()

    # ── Historial de mensajes ──────────────────────────────────────────────────
    agent_icons  = {"rag": "🕸️", "code": "💻", "researcher": "🔬"}
    agent_colors = {"rag": "violet", "code": "green", "researcher": "blue"}
    agent_names  = {"rag": "GraphRAG Agent", "code": "Code Agent", "researcher": "Researcher"}

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["query"])
        with st.chat_message("assistant"):
            a = entry["agent"]
            st.markdown(
                f":{agent_colors.get(a,'gray')}[**{agent_icons.get(a,'')} "
                f"{agent_names.get(a, a)}** — _{entry['reason']}_]"
            )
            st.markdown(entry["response"])

    # ── Input de chat ──────────────────────────────────────────────────────────
    typed = st.chat_input("Escribe tu pregunta aquí...")
    if typed:
        st.session_state.pending_query = typed

    # 
    # ── Procesar query ─────────────
    query = st.session_state.pending_query
    if query:
        st.session_state.pending_query = None

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            import time
            
            # 1. Capa de Caché en Memoria (Optimización de Costes) AQUI METEMEOS CACHE
            if "query_cache" not in st.session_state:
                st.session_state.query_cache = {}

            # Comprobamos si ya hemos respondido a esta pregunta exacta
            if query in st.session_state.query_cache:
                # CACHE HIT: Servimos la respuesta instantáneamente
                result = st.session_state.query_cache[query]
                st.success("⚡ Respuesta servida desde la caché local (0 tokens consumidos)")
                
            else:
                # CACHE MISS: Ejecutamos el pipeline completo con Streaming Visual
                with st.status("Iniciando arquitectura neuronal...", expanded=True) as status:
                    start_time = time.time()
                    wf = load_workflow()
                    
                    # Estado inicial, incluyendo el nuevo campo is_safe del guardrail
                    initial_state = {
                        "query": query, 
                        "agent_type": "", 
                        "agent_reason": "", 
                        "response": "",
                        "is_safe": True 
                    }
                    
                    # 2. Streaming de Nodos (Reducción de Latencia Percibida)
                    final_state = None
                    for output in wf.stream(initial_state):
                        for node_name, state in output.items():
                            st.write(f"✅ Nodo completado: **{node_name.upper()}**")
                            final_state = state # Guardamos el último estado procesado
                    
                    result = final_state
                    
                    # Guardamos en caché para futuras consultas idénticas
                    st.session_state.query_cache[query] = result
                    
                    elapsed_time = time.time() - start_time
                    status.update(
                        label=f"Proceso completado en {elapsed_time:.2f}s", 
                        state="complete", 
                        expanded=False
                    )

            # 3. Renderizado de la respuesta final
            a = result.get("agent_type", "unknown")
            reason = result.get("agent_reason", "Sin razón")
            response = result.get("response", "")
            is_safe = result.get("is_safe", True)

            # Verificamos la resolución del Guardrail
            if not is_safe:
                st.error("🛡️ **Bloqueo de Seguridad:** " + response)
            else:
                color = agent_colors.get(a, "gray")
                st.markdown(
                    f":{color}[**{agent_icons.get(a,'')} {agent_names.get(a, a)}** — _{reason}_]"
                )
                st.markdown(response)

            st.session_state.history.append(
                {"query": query, "agent": a, "reason": reason, "response": response}
            )

    # ── Limpiar historial ──────────────────────────────────────────────────────
    if st.session_state.history:
        st.divider()
        if st.button("🗑️ Limpiar conversación", type="secondary"):
            st.session_state.history = []
            st.session_state.query_cache = {} # Limpiamos la caché también
            st.rerun()
# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑA 2 — BASE DE CONOCIMIENTO
# ══════════════════════════════════════════════════════════════════════════════
with tab_kb:
    st.subheader("📚 Base de Conocimiento")

    st.markdown(
        """
        Esta base de conocimiento se creó **a mano** para esta práctica, escribiendo 15 fichas de texto.

        **En un proyecto real** la reemplazarías con documentos reales:
        - PDFs de documentación técnica
        - Páginas web descargadas
        - Artículos de investigación
        - Manuales de empresa

        El proceso sería exactamente igual: cargas los documentos con un `PyPDFLoader` o `WebBaseLoader`,
        los partes en trozos pequeños (*chunks*), los conviertes en vectores y los guardas en ChromaDB.
        El resto del sistema no cambiaría nada.
        """
    )

    st.divider()

    nodes = get_nodes()
    type_labels = {
        "concept":   "💡 Conceptos",
        "tool":      "🔧 Herramientas",
        "framework": "⚙️ Frameworks",
        "pattern":   "🧩 Patrones",
        "provider":  "🏢 Proveedores",
    }

    grouped: dict = defaultdict(list)
    for node in nodes:
        grouped[node["type"]].append(node)

    for type_key, group_nodes in grouped.items():
        st.markdown(f"#### {type_labels.get(type_key, type_key)}")
        cols = st.columns(min(len(group_nodes), 3))
        for i, node in enumerate(group_nodes):
            with cols[i % 3]:
                with st.expander(f"**{node['title']}**"):
                    st.caption(f"`id: {node['id']}` · `tipo: {node['type']}`")
                    st.write(node["content"])
        st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑA 3 — GRAFO
# ══════════════════════════════════════════════════════════════════════════════
with tab_graph:
    st.subheader("🕸️ Grafo de Conocimiento (NetworkX)")

    st.markdown(
        """
        El grafo conecta los 15 conceptos con **relaciones etiquetadas** (includes, uses, extends, is_a…).

        **¿Para qué sirve el grafo?**
        Cuando el **GraphRAG Agent** recibe una pregunta, hace dos cosas:
        1. Busca los 3 nodos más *similares* en ChromaDB (búsqueda vectorial)
        2. Recorre las aristas del grafo para traer también los *conceptos vecinos*

        Así la respuesta tiene más contexto que con RAG normal, aunque la pregunta no mencione esos conceptos.
        """
    )

    graph_path = Path(__file__).parent / "data" / "knowledge_graph.json"
    if graph_path.exists():
        fig = get_graph_figure()
        st.pyplot(fig)

        with st.expander("📋 Ver todas las relaciones del grafo"):
            with open(graph_path, encoding="utf-8") as f:
                data = json.load(f)
            import networkx as nx
            G = nx.node_link_graph(data)
            for src, dst, edge_data in sorted(G.edges(data=True), key=lambda x: x[0]):
                src_t = G.nodes[src].get("title", src)
                dst_t = G.nodes[dst].get("title", dst)
                st.markdown(f"- **{src_t}** → `{edge_data['relationship']}` → **{dst_t}**")
    else:
        st.error(
            "Grafo no encontrado. Ejecuta en la terminal:\n"
            "```bash\n"
            "uv run python scripts/build_knowledge_graph.py\n"
            "```"
        )

# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑA 4 — CÓMO FUNCIONA
# ══════════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.subheader("❓ Cómo funciona el sistema")

    st.markdown(
        """
        ### El flujo completo de una pregunta

        Cuando escribes una pregunta, esto es lo que pasa por dentro:

        ```
        Tu pregunta
             │
             ▼
        ┌──────────┐
        │  Router  │  ← ChatGPT lee la pregunta y decide a qué agente enviarla
        └──────────┘
             │
             ├── "rag"        ──► 🕸️  GraphRAG Agent  (busca en la BD de conocimiento)
             ├── "code"       ──► 💻  Code Agent       (genera código Python)
             └── "researcher" ──► 🔬  Researcher Agent  (responde con conocimiento general)
                                        │
                                        ▼
                                  Respuesta final
        ```

        ---

        ### Las piezas del sistema

        #### 📁 `prompts/` — Las instrucciones para cada agente
        Cada agente tiene un *system prompt* diferente. Es lo primero que ve ChatGPT antes de leer
        tu pregunta. Define la personalidad y las reglas del agente. Están separados del código
        para poder mejorarlos sin tocar lógica.

        #### 📁 `agents/` — El código de cada agente
        Cada archivo es una función que llama a `ChatOpenAI` (la API de ChatGPT) con su prompt.
        El decorador `@traceable` hace que LangSmith registre esa llamada como un *span* en la traza.

        #### 📁 `rag/` — La recuperación de información (GraphRAG)
        - **`retriever.py`**: abre la conexión con ChromaDB. Es como el conector a la base de datos.
        - **`graph_rag.py`**: la lógica completa. Busca vectorialmente, expande por el grafo, y devuelve el contexto enriquecido al agente RAG.

        #### 📁 `graph/` — El flujo orquestado con LangGraph
        `workflow.py` define el *grafo de estados*: qué nodo va después de cuál.
        El estado es un diccionario que se va completando a lo largo del flujo:
        ```python
        estado = {
            "query": "¿Qué es LangSmith?",     # tu pregunta
            "agent_type": "rag",                # lo que decide el router
            "agent_reason": "es una herram...", # por qué eligió ese agente
            "response": "LangSmith es..."       # la respuesta final
        }
        ```

        #### 📁 `scripts/` — Scripts de preparación (se ejecutan UNA SOLA VEZ)
        - `build_knowledge_graph.py` → crea `data/knowledge_graph.json`
        - `build_vector_db.py` → crea `data/chroma_db/` con los vectores (embeddings de OpenAI)

        #### 📁 `data/` — Los datos persistidos
        - `knowledge_base.py` → los 15 textos escritos a mano (podrían ser PDFs)
        - `knowledge_graph.json` → el grafo (nodos + aristas) en formato JSON
        - `chroma_db/` → la base de datos vectorial (archivos binarios de ChromaDB)

        ---

        ### ¿Qué registra LangSmith de cada petición?

        Para cada pregunta que haces, LangSmith guarda una traza con esta jerarquía:

        ```
        🔵 [chain] router             ← llama al router
          └─ 🟣 [llm] ChatOpenAI     ← llamada real a gpt-4o-mini

        🟢 [retriever] rag_agent      ← (si fue al agente RAG)
          └─ 🟣 [llm] ChatOpenAI     ← llamada real a gpt-4o-mini
        ```

        Puedes ver el prompt exacto que se envió, la respuesta, los tokens usados y la latencia.
        """
    )
