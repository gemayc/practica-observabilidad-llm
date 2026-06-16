# ─── Workflow: el grafo de estados que orquesta todos los agentes ──────────────
# Este archivo define el FLUJO COMPLETO del sistema usando LangGraph.
#
# ¿Qué es un StateGraph?
#   Es un grafo donde cada nodo es una función Python y cada arista es una transición.
#   Hay un "estado" compartido (AgentState) que se va completando a lo largo del flujo.
#   El estado empieza con la pregunta y termina con la respuesta del agente elegido.
#
# El flujo que construimos aquí es:
#
#   START
#     │
#     ▼
#   router  ← lee la pregunta, decide el agente
#     │
#     ├── "researcher" ──► researcher ──► END
#     ├── "code"       ──► code       ──► END
#     └── "rag"        ──► rag        ──► END

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

# Importamos los agentes que serán los nodos del grafo
from agents.code_agent import run_code_agent
from agents.rag_agent import run_rag_agent
from agents.researcher_agent import run_researcher
from agents.router import route_query
from agents.guardrail_agent import run_guardrail


# ── Estado compartido del grafo ────────────────────────────────────────────────
# TypedDict define la estructura del diccionario que viaja entre nodos.
# Cada nodo recibe este estado, lo lee, lo modifica y lo devuelve.
class AgentState(TypedDict):
    query: str          # La pregunta original del usuario (no cambia)
    agent_type: str     # Lo rellena el router: "researcher", "code" o "rag"
    agent_reason: str   # La razón que dio el router para elegir ese agente
    response: str       # Lo rellena el agente elegido: la respuesta final
    is_safe: bool       # <-- viene del agente guardrail: si la pregunta es segura o no


# ── Nodos del grafo (cada uno es una función que transforma el estado) ─────────

#Creamos el nuevo nodo
def guardrail_node(state: AgentState) -> AgentState:
    """Nodo de seguridad: verifica si el prompt es malicioso."""
    result = run_guardrail(state["query"])
    
    # Si no es seguro, ya dejamos una respuesta por defecto preparada
    safe_response = "" if result["is_safe"] else "Lo siento, no puedo procesar esta solicitud por motivos de seguridad."
    
    return {**state, "is_safe": result["is_safe"], "response": safe_response}

def router_node(state: AgentState) -> AgentState:
    """Nodo router: llama a route_query() y guarda en el estado qué agente usar."""
    result = route_query(state["query"])
    # Devolvemos el estado actualizado con el tipo de agente y la razón de la elección
    return {**state, "agent_type": result["agent"], "agent_reason": result["reason"]}


def researcher_node(state: AgentState) -> AgentState:
    """Nodo researcher: llama al agente de preguntas conceptuales y guarda la respuesta."""
    return {**state, "response": run_researcher(state["query"])}


def code_node(state: AgentState) -> AgentState:
    """Nodo code: llama al agente generador de código y guarda la respuesta."""
    return {**state, "response": run_code_agent(state["query"])}


def rag_node(state: AgentState) -> AgentState:
    """Nodo rag: llama al agente GraphRAG (busca en BD + responde) y guarda la respuesta."""
    return {**state, "response": run_rag_agent(state["query"])}

#  Nueva función de enrutamiento condicional inicial
def check_safety(state: AgentState) -> Literal["router", "unsafe_end"]:
    """Decide si vamos al router normal o si abortamos la ejecución."""
    if state.get("is_safe", True):
        return "router"
    return "unsafe_end"


# ── Función de enrutamiento condicional ────────────────────────────────────────
# Esta función lee el estado DESPUÉS del router y decide a qué nodo ir a continuación.
# LangGraph la usa para saber qué arista tomar.
def select_agent(state: AgentState) -> Literal["researcher", "code", "rag"]:
    """Devuelve el nombre del siguiente nodo según lo que decidió el router."""
    return state["agent_type"]  # type: ignore[return-value]


# ── Construcción y compilación del grafo ───────────────────────────────────────
def build_workflow():
    """
    Construye el grafo LangGraph con todos los nodos y aristas, y lo compila.
    Devuelve un objeto 'app' al que se le puede llamar con .invoke(estado_inicial).
    """

    # Creamos el grafo indicando el tipo del estado compartido
    workflow = StateGraph(AgentState)

    # Registramos cada nodo con un nombre y la función que lo implementa
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("router", router_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("code", code_node)
    workflow.add_node("rag", rag_node)

    #  Modificamos las aristas (edges)
    # Ahora el flujo empieza obligatoriamente en el guardrail no en el router
    workflow.add_edge(START, "guardrail")
    
    # Del guardrail sale una decisión: o vamos al router, o terminamos (END)
    workflow.add_conditional_edges(
        "guardrail",
        check_safety,
        {
            "router": "router",
            "unsafe_end": END, # Si es inseguro, termina directamente
        },
    )

    # Arista condicional: desde el router, va a researcher, code o rag
    # según lo que devuelva la función select_agent()
    workflow.add_conditional_edges(
        "router",           # nodo de origen
        select_agent,       # función que decide el destino
        {                   # mapa de posibles destinos
            "researcher": "researcher",
            "code": "code",
            "rag": "rag",
        },
    )

    # Aristas fijas: cada agente termina en END (fin del flujo)
    workflow.add_edge("researcher", END)
    workflow.add_edge("code", END)
    workflow.add_edge("rag", END)

    # compile() valida el grafo y devuelve un objeto ejecutable
    return workflow.compile()
