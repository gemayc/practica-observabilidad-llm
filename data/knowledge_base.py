# ─── Base de Conocimiento: los datos que alimentan el sistema RAG ──────────────
# Este archivo contiene los textos sobre el tema del proyecto:
# Observabilidad en sistemas LLM.
#
# En un proyecto real, estos textos vendrían de:
#   - PDFs cargados con PyPDFLoader
#   - Páginas web descargadas con WebBaseLoader
#   - Documentación técnica en Markdown
#
# Aquí los escribimos manualmente para tener control total sobre el contenido.
#
# Estructura de cada nodo (NODES):
#   - id:      identificador único, también se usa en el grafo
#   - type:    categoría del concepto (concept / tool / framework / pattern / provider)
#   - title:   nombre legible del concepto
#   - content: texto completo que se vectoriza y guarda en ChromaDB
#
# Estructura de cada arista (EDGES):
#   Tupla de 3 elementos: (nodo_origen, nodo_destino, relacion)
#   Ejemplo: ("rag", "vector_db", "usa") → "RAG usa una base de datos vectorial"

# ── 15 nodos de conocimiento sobre Observabilidad LLM ─────────────────────────
NODES = [
    {
        "id": "observability",
        "type": "concept",
        "title": "Observabilidad en LLMs",
        "content": (
            "La observabilidad en LLMs es la práctica de monitorizar, trazar y evaluar aplicaciones "
            "de Modelos de Lenguaje en producción. Abarca el seguimiento de entradas y salidas, "
            "la latencia, el uso de tokens, los costes y el comportamiento del modelo. Sus componentes "
            "clave incluyen el rastreo distribuido, los frameworks de evaluación y los paneles de "
            "monitorización en tiempo real. La observabilidad ayuda a los ingenieros a depurar problemas, "
            "optimizar el rendimiento y garantizar la fiabilidad de los sistemas de IA. Se diferencia de "
            "la observabilidad de software tradicional en que las salidas de los LLMs son no deterministas, "
            "por lo que es esencial capturar los pares prompt/respuesta."
        ),
    },
    {
        "id": "langsmith",
        "type": "tool",
        "title": "LangSmith",
        "content": (
            "LangSmith es una plataforma de desarrollo creada por LangChain para depurar, probar, evaluar "
            "y monitorizar aplicaciones LLM. Proporciona rastreo distribuido que captura el árbol de "
            "ejecución completo de las ejecuciones de LangChain, incluyendo entradas, salidas y pasos "
            "intermedios. LangSmith permite crear conjuntos de datos para evaluación, pipelines de pruebas "
            "automatizadas y monitorización en producción con alertas. La integración se realiza configurando "
            "las variables de entorno LANGCHAIN_TRACING_V2=true y LANGCHAIN_API_KEY. El decorador @traceable "
            "envuelve cualquier función Python para crear spans personalizados."
        ),
    },
    {
        "id": "tracing",
        "type": "concept",
        "title": "Rastreo Distribuido",
        "content": (
            "El rastreo distribuido en aplicaciones LLM captura la ruta de ejecución completa de una "
            "petición a través de múltiples componentes. Una traza contiene múltiples spans, donde cada "
            "span representa una operación individual (llamada a LLM, uso de herramienta, paso de "
            "recuperación). Las trazas permiten depurar mostrando exactamente qué ocurrió en cada paso: "
            "los prompts enviados, las respuestas recibidas, la latencia en cada nodo y el consumo de "
            "tokens. LangSmith y OpenTelemetry son herramientas habituales para el rastreo de LLMs. "
            "Las relaciones padre-hijo entre spans revelan la jerarquía de llamadas en flujos de agentes complejos."
        ),
    },
    {
        "id": "evaluation",
        "type": "concept",
        "title": "Evaluación de LLMs",
        "content": (
            "La evaluación de LLMs es la valoración sistemática del rendimiento de los modelos de lenguaje. "
            "Incluye la evaluación offline usando conjuntos de datos curados con respuestas de referencia, "
            "la evaluación online monitorizando el tráfico en vivo, y los enfoques LLM-como-juez donde un "
            "modelo evalúa a otro. Las métricas clave incluyen exactitud, fidelidad (para RAG), relevancia "
            "de la respuesta y toxicidad. LangSmith proporciona evaluadores integrados y soporta pipelines "
            "de evaluación personalizados con recopilación de feedback humano. Los conjuntos de datos de "
            "evaluación pueden construirse a partir de trazas de producción marcando ejemplos interesantes."
        ),
    },
    {
        "id": "monitoring",
        "type": "concept",
        "title": "Monitorización en Producción",
        "content": (
            "La monitorización en producción para aplicaciones LLM sigue métricas operacionales en tiempo "
            "real. Las señales importantes incluyen el volumen de peticiones, las tasas de error, la "
            "latencia p50/p95/p99, el uso de tokens por petición, el coste por operación y las puntuaciones "
            "de feedback de los usuarios. La detección de anomalías marca cambios repentinos en la "
            "distribución de salidas. El panel de monitorización de LangSmith proporciona información en "
            "tiempo real y permite configurar alertas para violaciones de umbrales. La monitorización debe "
            "desglosarse por modelo, por versión de plantilla de prompt y por segmento de usuario."
        ),
    },
    {
        "id": "langchain",
        "type": "framework",
        "title": "LangChain",
        "content": (
            "LangChain es un framework de código abierto para construir aplicaciones potenciadas por LLMs. "
            "Proporciona abstracciones para LLMs, modelos de chat, modelos de embeddings, almacenes de "
            "vectores, cargadores de documentos y cadenas. LCEL (LangChain Expression Language) permite "
            "pipelines componibles usando el operador pipe. LangChain se integra con más de 100 proveedores "
            "de LLMs y herramientas, facilitando la construcción de chatbots, sistemas RAG y agentes "
            "autónomos. Envía trazas automáticamente a LangSmith cuando se configuran las variables de "
            "entorno de rastreo, sin necesidad de cambios en el código."
        ),
    },
    {
        "id": "langgraph",
        "type": "framework",
        "title": "LangGraph",
        "content": (
            "LangGraph es una librería construida sobre LangChain para crear aplicaciones con estado y "
            "múltiples actores usando flujos de trabajo basados en grafos. Define los agentes como nodos "
            "en un grafo dirigido donde las aristas representan transiciones entre estados. LangGraph "
            "soporta ciclos, permitiendo que los agentes iteren y reintenten. Conceptos clave: AgentState "
            "(TypedDict para el estado compartido), nodos (funciones Python que actualizan el estado), "
            "aristas (transiciones entre nodos), aristas condicionales (lógica de enrutamiento mediante "
            "funciones) y checkpointing para persistencia y human-in-the-loop. Todas las ejecuciones del "
            "grafo se rastrean automáticamente en LangSmith."
        ),
    },
    {
        "id": "rag",
        "type": "pattern",
        "title": "Generación Aumentada por Recuperación (RAG)",
        "content": (
            "La Generación Aumentada por Recuperación (RAG) es un patrón que mejora las respuestas de los "
            "LLMs recuperando contexto relevante de una base de conocimiento antes de generar una respuesta. "
            "El pipeline tiene tres pasos: indexación (dividir documentos en trozos, convertirlos en "
            "vectores y almacenarlos en una BD vectorial), recuperación (convertir la consulta en vector "
            "y encontrar documentos similares por similitud coseno) y generación (combinar el contexto "
            "recuperado con la consulta en un prompt). RAG reduce las alucinaciones basando las respuestas "
            "en hechos recuperados y permite a los LLMs acceder a información más allá de su fecha de "
            "corte de entrenamiento. LangSmith rastrea cada paso de recuperación por separado."
        ),
    },
    {
        "id": "graph_rag",
        "type": "pattern",
        "title": "GraphRAG",
        "content": (
            "GraphRAG (Generación Aumentada por Recuperación con Grafo) extiende el RAG tradicional "
            "incorporando una estructura de grafo de conocimiento. En lugar de tratar los documentos como "
            "trozos aislados, GraphRAG construye un grafo donde los nodos representan entidades o conceptos "
            "y las aristas representan relaciones. Durante la recuperación, combina la búsqueda por "
            "similitud vectorial con el recorrido del grafo para encontrar no solo los documentos "
            "directamente relevantes, sino también los conceptos relacionados a través de los vecinos "
            "del grafo. Esto proporciona información más rica y contextual que el RAG vectorial puro. "
            "Microsoft Research publicó el artículo fundacional sobre este enfoque. NetworkX se usa "
            "habitualmente para la estructura del grafo."
        ),
    },
    {
        "id": "vector_db",
        "type": "concept",
        "title": "Base de Datos Vectorial",
        "content": (
            "Una base de datos vectorial almacena embeddings vectoriales de alta dimensión y soporta "
            "búsqueda eficiente por similitud usando algoritmos como HNSW (Hierarchical Navigable Small "
            "World) o IVF (Inverted File Index). A diferencia de las bases de datos tradicionales que "
            "buscan valores exactos, las bases de datos vectoriales encuentran elementos semánticamente "
            "similares calculando la similitud coseno o la distancia euclídea entre vectores de embedding. "
            "Son el núcleo de los sistemas RAG, la búsqueda semántica y los motores de recomendación. "
            "Las opciones más populares incluyen ChromaDB, Pinecone, Weaviate, Qdrant y pgvector."
        ),
    },
    {
        "id": "chromadb",
        "type": "tool",
        "title": "ChromaDB",
        "content": (
            "ChromaDB es una base de datos vectorial de código abierto y nativa para IA, diseñada para "
            "la simplicidad y la experiencia del desarrollador. Soporta modos de almacenamiento en memoria "
            "y persistente, siendo ideal tanto para prototipado como para producción. ChromaDB proporciona "
            "un cliente Python con generación automática de embeddings, filtrado por metadatos y soporte "
            "multimodal. Se integra de forma nativa con LangChain mediante el paquete langchain-chroma. "
            "Las colecciones almacenan documentos con embeddings, metadatos e IDs, soportando operaciones "
            "de añadir, actualizar, eliminar y consultar por similitud. La función de embedding por defecto "
            "usa all-MiniLM-L6-v2 localmente."
        ),
    },
    {
        "id": "agent",
        "type": "pattern",
        "title": "Agente LLM",
        "content": (
            "Un agente LLM es un sistema autónomo que usa un modelo de lenguaje para decidir qué acciones "
            "tomar para completar una tarea. El bucle central es: observar (recibir entrada), pensar "
            "(razonar sobre qué hacer), actuar (llamar a una herramienta o producir una salida), repetir. "
            "Los agentes pueden usar herramientas como búsqueda web, ejecución de código, consultas a bases "
            "de datos y APIs. ReAct (Reasoning + Acting) es una arquitectura de agente común que intercala "
            "trazas de razonamiento con llamadas a acciones. Los sistemas multi-agente tienen agentes "
            "especializados que colaboran, a menudo orquestados por un agente router. LangGraph es el "
            "framework preferido para flujos de trabajo multi-agente complejos."
        ),
    },
    {
        "id": "prompt_engineering",
        "type": "concept",
        "title": "Ingeniería de Prompts",
        "content": (
            "La ingeniería de prompts es la práctica de diseñar entradas efectivas para los modelos de "
            "lenguaje con el fin de obtener las salidas deseadas. Técnicas clave: prompting zero-shot "
            "(instrucción directa), prompting few-shot (ejemplos en el prompt), cadena de pensamiento "
            "(pedir razonamiento paso a paso), system prompts (establecer la persona y las restricciones "
            "del modelo) y prompting de salida estructurada (esquemas JSON/XML). Los buenos prompts son "
            "claros, específicos y aportan el contexto necesario. LangSmith ayuda a evaluar la efectividad "
            "de los prompts mediante pruebas A/B sobre conjuntos de datos y el seguimiento de la calidad "
            "de las salidas a lo largo del tiempo. El versionado de prompts es una buena práctica para "
            "sistemas en producción."
        ),
    },
    {
        "id": "openai",
        "type": "provider",
        "title": "OpenAI",
        "content": (
            "OpenAI proporciona los modelos GPT-4o, GPT-4o-mini, o1 y o3 mediante API. La API de Chat "
            "Completions acepta una lista de mensajes y devuelve una respuesta. Parámetros clave: model, "
            "messages, temperature (0=determinista, 1=creativo), max_tokens y response_format para salida "
            "en JSON. El uso de herramientas (function calling) permite a los LLMs invocar funciones "
            "externas. El paquete langchain-openai envuelve la API con las abstracciones de LangChain y "
            "rastreo automático en LangSmith. El precio se cobra por millón de tokens (entrada y salida "
            "se cobran por separado). gpt-4o-mini ofrece la mejor relación coste/rendimiento para la "
            "mayoría de las aplicaciones."
        ),
    },
    {
        "id": "metrics",
        "type": "concept",
        "title": "Métricas LLM",
        "content": (
            "Métricas clave para sistemas LLM: Latencia (tiempo hasta el primer token, tiempo total de "
            "generación), Rendimiento (peticiones por segundo), Uso de tokens (tokens de prompt, tokens "
            "de respuesta, total), Coste (euros por petición), Puntuaciones de calidad (fidelidad, "
            "relevancia, coherencia para RAG), Tasa de error (peticiones fallidas, rechazos) y Feedback "
            "de usuarios (me gusta/no me gusta, valoraciones). Estas métricas deben seguirse por modelo, "
            "por versión de plantilla de prompt y por segmento de usuario. LangSmith agrega estas métricas "
            "automáticamente de las ejecuciones rastreadas y proporciona paneles con gráficos de series "
            "temporales. Configurar alertas de coste evita picos de facturación inesperados."
        ),
    },
]

# ── 21 aristas: relaciones entre los conceptos del grafo ─────────────────────
# Formato: (nodo_origen, nodo_destino, tipo_de_relacion)
# Estas relaciones se usan en GraphRAG para expandir el contexto de búsqueda
EDGES = [
    # Observabilidad engloba trazas, evaluación y monitoreo
    ("observability", "tracing",    "incluye"),
    ("observability", "evaluation", "incluye"),
    ("observability", "monitoring", "incluye"),
    ("observability", "metrics",    "mide"),

    # LangSmith es la herramienta principal para cada aspecto de la observabilidad
    ("langsmith", "tracing",    "proporciona"),
    ("langsmith", "evaluation", "proporciona"),
    ("langsmith", "monitoring", "proporciona"),
    ("langsmith", "langchain",  "parte_de"),

    # LangGraph extiende LangChain y permite orquestar agentes
    ("langgraph", "langchain", "extiende"),
    ("langgraph", "agent",     "orquesta"),

    # LangChain habilita agentes y facilita RAG
    ("langchain", "agent", "habilita"),
    ("langchain", "rag",   "implementa"),

    # RAG usa base de datos vectorial; GraphRAG extiende RAG añadiendo el grafo
    ("rag",       "vector_db", "usa"),
    ("graph_rag", "rag",       "extiende"),
    ("graph_rag", "vector_db", "usa"),

    # ChromaDB es una implementación concreta de base de datos vectorial
    ("chromadb", "vector_db", "es_un"),

    # Los agentes usan prompt engineering y están impulsados por OpenAI
    ("agent", "prompt_engineering", "usa"),
    ("agent", "openai",             "impulsado_por"),

    # RAG también usa OpenAI para generar los embeddings y las respuestas
    ("rag", "openai", "usa"),

    # Las métricas alimentan el sistema de monitoreo
    ("metrics",    "monitoring", "alimenta"),
    ("evaluation", "metrics",    "produce"),
]
