<div align="center">

# 🤖 LLM Observability Assistant

### Arquitectura Multi-Agente con GraphRAG, LangGraph y Evaluación Continua

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-FF6B35?style=for-the-badge&logo=langchain&logoColor=white)](https://smith.langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-E35B5B?style=for-the-badge)](https://www.trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)

</div>

---

## 📖 Descripción

Este proyecto implementa una **arquitectura multi-agente avanzada** centrada en la observabilidad, el control de flujo y la evaluación continua de Modelos de Lenguaje Grande (LLMs).

En lugar de depender de una única llamada monolítica a un LLM, el sistema orquesta diferentes **agentes especialistas** mediante un grafo de estados (LangGraph), garantizando seguridad perimetral, eficiencia en costes y respuestas enriquecidas mediante recuperación basada en grafos (GraphRAG).

Todo el ciclo de vida —desde la consulta hasta la evaluación— queda trazado en **LangSmith**, proporcionando visibilidad completa sobre latencia, consumo de tokens y calidad de las respuestas.

---

## 🏗️ Decisiones de Arquitectura

### Orquestación de Flujo — LangGraph

El sistema utiliza un **enrutador semántico** para derivar el tráfico al agente correspondiente (GraphRAG, Code o Researcher). Esto aísla responsabilidades y evita que un único prompt monolítico colapse ante consultas heterogéneas.

### GraphRAG — NetworkX + ChromaDB

Se sustituye el RAG tradicional (recuperación plana) por un enfoque de grafos:

- **ChromaDB** realiza la búsqueda vectorial en `O(1)` para localizar los 3 conceptos más próximos semánticamente.
- **NetworkX** en memoria RAM expande el contexto hacia los nodos vecinos del grafo de conocimiento, añadiendo relaciones implícitas sin saturar la ventana de contexto.

> **¿Por qué?** Para reducir el ruido en el contexto, minimizar el consumo de tokens y conectar ideas lógicamente sin exponer al modelo a información irrelevante.

### Cortafuegos Perimetral — Guardrail

El **primer nodo** del grafo es un filtro de seguridad estricto que intercepta inyecciones de prompts o consultas maliciosas *antes* de que consuman recursos de base de datos o tiempo de cómputo en los agentes especialistas. Patrón **Fail Fast**.

### Optimización de Latencia y Costes

| Técnica | Beneficio |
|---|---|
| **Caché en Memoria** | Consultas repetidas resueltas en `O(1)` sin invocar la API del LLM — ahorro del 100% en tokens |
| **Streaming de Nodos** | Retransmisión de telemetría del grafo en tiempo real hacia la UI, reduciendo la latencia percibida |
| **Carga Perezosa** | ChromaDB y NetworkX se inicializan solo en el primer uso, reduciendo el tiempo de arranque |

---

## 🔬 Observabilidad y DevOps para IA

El ciclo de vida del sistema se controla mediante dos capas de evaluación diferenciadas:

### 1. Evaluación Online — Producción con LangSmith

Trazabilidad completa de cada llamada al LLM. LangSmith registra automáticamente:

- Latencia por nodo (P50 / P99)
- Coste por token en cada agente
- Puntuación de relevancia mediante **LLM-as-a-Judge**
- Árbol de ejecución completo del grafo

### 2. Evaluación Offline — Pipeline CI/CD Local

Scripts para validar la solidez de la infraestructura antes de desplegar a producción:

| Script | Tipo | Qué mide |
|---|---|---|
| `evaluate_dataset.py` | **Golden Set Test** | Enrutamiento correcto y seguridad (macro) |
| `evaluate_triad.py` | **RAG Triad Test** | Context Relevance · Faithfulness · Answer Relevance (micro) |

---

## 🗺️ Flujo del Sistema

```
Consulta del Usuario
        │
        ▼
┌───────────────────────┐
│   🛡️  GUARDRAIL AGENT  │  ← Detecta inyecciones y toxicidad
└──────────┬────────────┘
           │
    ┌──────▼──────┐
    │  ¿Es seguro? │
    └──┬───────┬──┘
       │ NO    │ SÍ
       ▼       ▼
   Bloquear  ┌─────────────────────┐
             │   🧭 ROUTER AGENT   │  ← Clasifica la consulta
             └──────────┬──────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   ┌─────────────┐ ┌──────────┐ ┌──────────────┐
   │ 🔬 Research  │ │ 💻 Code  │ │ 🕸️  GraphRAG  │
   │    Agent    │ │  Agent   │ │    Agent     │
   └─────────────┘ └──────────┘ └──────────────┘
          │             │             │
          └─────────────┴─────────────┘
                        │
                        ▼
               Respuesta al Usuario
                        │
                        ▼
               📊 LangSmith Trace
```

---

## 📁 Estructura del Proyecto

```
practica-observabilidad-llm/
│
├── 🤖 agents/                      # Implementación de los agentes
│   ├── guardrail_agent.py          # Filtro de seguridad perimetral
│   ├── router.py                   # Clasificador semántico de consultas
│   ├── researcher_agent.py         # Agente de conocimiento conceptual
│   ├── code_agent.py               # Agente generador de código Python
│   └── rag_agent.py                # Agente de búsqueda en base de conocimiento
│
├── 📐 graph/
│   └── workflow.py                 # Grafo de estados LangGraph (orquestador)
│
├── 🔍 rag/
│   ├── graph_rag.py                # GraphRAG: búsqueda vectorial + expansión por grafo
│   └── retriever.py                # Conector ChromaDB
│
├── 📝 prompts/                     # System prompts por agente
│   ├── router_prompt.py
│   ├── researcher_prompt.py
│   ├── code_agent_prompt.py
│   └── rag_prompt.py
│
├── 🗄️  data/
│   ├── knowledge_base.py           # 15 nodos y 21 relaciones del grafo de conocimiento
│   ├── knowledge_graph.json        # Grafo serializado (generado por scripts)
│   └── chroma_db/                  # Base de datos vectorial persistente (generada)
│
├── ⚙️  scripts/                     # Scripts de preparación y evaluación
│   ├── build_knowledge_graph.py    # Construye y serializa el grafo NetworkX
│   ├── build_vector_db.py          # Vectoriza nodos y puebla ChromaDB
│   ├── evaluate_dataset.py         # Evaluación Golden Set contra LangSmith
│   └── evaluate_triad.py           # Evaluación RAG Triad (3 métricas de calidad)
│
├── 🖥️  main.py                      # Interfaz CLI (Rich, terminal interactivo)
├── 🌐 app.py                        # Interfaz Web (Streamlit, 4 pestañas)
├── pyproject.toml                  # Metadatos y dependencias del proyecto
└── .env                            # Variables de entorno (no commitear)
```

---

## 🧠 Agentes Especialistas

### 🛡️ Guardrail Agent
Primer nodo del grafo. Usa `gpt-4o-mini` con `temperature=0` para una detección **determinista** de prompts maliciosos e inyecciones. Si la consulta no es segura, el grafo termina inmediatamente sin invocar ningún agente especialista.

### 🧭 Router Agent
Clasifica cada consulta y decide qué agente debe responder, con el siguiente orden de prioridad:

1. **RAG** — si la consulta menciona observabilidad, LangSmith, LangChain, RAG, métricas u otros conceptos de la base de conocimiento.
2. **Code** — si se solicita explícitamente generación o explicación de código.
3. **Researcher** — para preguntas conceptuales o comparativas generales.

### 🔬 Researcher Agent
Responde preguntas teóricas y conceptuales sobre IA, LLMs y MLOps utilizando el conocimiento del propio modelo. `temperature=0.3` para respuestas naturales y explicativas.

### 💻 Code Agent
Genera código Python completo, moderno (3.10+, type hints) y ejecutable sobre el ecosistema AI/ML (LangChain, LangGraph, OpenAI). `temperature=0.1` para máxima precisión.

### 🕸️ GraphRAG Agent
El agente más sofisticado. Combina búsqueda vectorial (ChromaDB) con expansión por grafo (NetworkX) para recuperar contexto enriquecido antes de generar la respuesta. El contexto recuperado incluye nodos directos *y* sus vecinos en el grafo de conocimiento.

---

## 🕸️ Base de Conocimiento

El grafo de conocimiento está formado por **15 nodos** y **21 relaciones dirigidas**, cubriendo el dominio de observabilidad en LLMs:

| Categoría | Nodos |
|---|---|
| **Conceptos** | Observabilidad en LLMs · Rastreo Distribuido · Evaluación de LLMs · Monitorización en Producción · Ingeniería de Prompts · Métricas LLM · Base de Datos Vectorial |
| **Frameworks** | LangChain · LangGraph |
| **Patrones** | RAG · GraphRAG |
| **Herramientas** | LangSmith · ChromaDB |
| **Proveedores** | OpenAI |
| **Agentes** | Agente LLM |

**Ejemplos de relaciones:** `observabilidad → incluye → rastreo` · `langsmith → proporciona → evaluacion` · `graph_rag → extiende → rag` · `rag → usa → vector_db`

---

## ⚙️ Instalación

### Requisitos Previos

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (gestor de paquetes recomendado)
- Cuenta en [OpenAI](https://platform.openai.com/) con API Key
- Cuenta en [LangSmith](https://smith.langchain.com/) con API Key (opcional, para trazabilidad)

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd practica-observabilidad-llm
```

### 2. Instalar dependencias

```bash
uv sync
```

### 3. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
# Obligatorio
OPENAI_API_KEY=sk-proj-...

# Opcional — activa trazabilidad completa en LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT="practica-observabilidad-llm"
LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com
```

### 4. Construir la base de conocimiento

> ⚠️ Este paso solo es necesario la primera vez (o si modificas `data/knowledge_base.py`).

```bash
# Construye el grafo de conocimiento NetworkX
uv run python scripts/build_knowledge_graph.py

# Vectoriza los nodos y puebla ChromaDB
uv run python scripts/build_vector_db.py
```

---

## 🚀 Uso

### Interfaz Web — Streamlit

```bash
uv run streamlit run app.py
```

La interfaz se abre en `http://localhost:8501` y ofrece **4 pestañas**:

| Pestaña | Contenido |
|---|---|
| 💬 **Chat** | Conversación interactiva con streaming de nodos y caché inteligente |
| 📚 **Knowledge Base** | Explorador de los 15 nodos del grafo por categoría |
| 🕸️ **Graph Visualization** | Visualización del grafo de conocimiento con NetworkX + Matplotlib |
| ❓ **How It Works** | Documentación interactiva del flujo y componentes del sistema |

### Interfaz CLI — Terminal

```bash
uv run python main.py
```

Terminal interactivo con salida enriquecida (Rich). Muestra el agente seleccionado, el motivo del enrutamiento y la respuesta formateada en Markdown. Escribe `salir`, `exit` o `quit` para terminar.

---

## 📊 Evaluación

### Golden Set Test — Enrutamiento y Seguridad

Evalúa el comportamiento macro del sistema contra un conjunto de preguntas etiquetadas en LangSmith. Un juez LLM (`gpt-4o-mini`) puntúa cada respuesta como correcta o incorrecta.

```bash
uv run python scripts/evaluate_dataset.py
```

> Requiere tener el dataset `golden-set-observabilidad` creado en LangSmith.

### RAG Triad Test — Calidad de la Recuperación

Evalúa la calidad de la capa RAG con tres métricas complementarias (escala 1-5):

| Métrica | Pregunta que responde |
|---|---|
| **Faithfulness** | ¿La respuesta es fiel al contexto recuperado? |
| **Answer Relevance** | ¿La respuesta aborda realmente la pregunta del usuario? |
| **Context Relevance** | ¿GraphRAG recuperó información verdaderamente útil? |

```bash
uv run python scripts/evaluate_triad.py
```

---

## 🔭 Observabilidad con LangSmith

Con LangSmith activado, cada consulta genera un árbol de trazas completo:

```
🔵 [chain]     input_guardrail
🔵 [chain]     router
  └─ 🟣 [llm]  ChatOpenAI (gpt-4o-mini)
🟢 [retriever] rag_agent
  └─ 🟣 [llm]  ChatOpenAI (gpt-4o-mini)
```

Cada traza captura automáticamente:

- **Prompt enviado** y **respuesta recibida**
- **Modelo** y **temperatura** usados
- **Tokens consumidos** (entrada / salida / total)
- **Latencia** por nodo y latencia total de extremo a extremo
- **Errores y excepciones** con stack trace completo

---

## 🛠️ Stack Tecnológico

| Categoría | Tecnología | Rol |
|---|---|---|
| **Orquestación** | LangGraph | Grafo de estados multi-agente |
| **LLM** | OpenAI GPT-4o-mini | Backbone de todos los agentes |
| **Embeddings** | OpenAI text-embedding-3-small | Vectorización de la base de conocimiento |
| **Vector DB** | ChromaDB | Búsqueda semántica de alta velocidad |
| **Graph DB** | NetworkX | Expansión por relaciones en memoria RAM |
| **Framework** | LangChain | Abstracciones de agentes y cadenas |
| **Observabilidad** | LangSmith | Trazabilidad, métricas y evaluación |
| **Web UI** | Streamlit | Interfaz web de 4 pestañas |
| **CLI UI** | Rich | Terminal interactivo con color |
| **Visualización** | Matplotlib | Renderizado del grafo de conocimiento |

---

## 📄 Licencia

Este proyecto es de uso educativo. Desarrollado como práctica de observabilidad y arquitectura multi-agente con LLMs.

---

<div align="center">

Desarrollado con ❤️ como práctica de **Observabilidad en LLMs**

</div>
