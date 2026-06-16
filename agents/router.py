# ─── Router: decide a qué agente enviar la pregunta del usuario ───────────────
# Este archivo es el "director de tráfico" del sistema.
# Recibe la pregunta, se la manda a ChatGPT con instrucciones especiales,
# y ChatGPT responde con el nombre del agente más adecuado en formato JSON.

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

# Importamos los dos textos del prompt (sistema + humano) desde la carpeta prompts/
from prompts.router_prompt import ROUTER_HUMAN, ROUTER_SYSTEM


# @traceable hace que LangSmith registre cada llamada a esta función como un "span"
# run_type="chain" indica que es un paso de orquestación (no directamente un LLM)
@traceable(name="router", run_type="chain")
def route_query(query: str) -> dict[str, str]:
    """Recibe la pregunta del usuario y devuelve {'agent': ..., 'reason': ...}."""

    # temperature=0 → respuesta determinista (siempre el mismo JSON, sin variación creativa)
    # Necesario porque el router DEBE devolver JSON válido, no texto libre
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Construimos la conversación: primero el rol del sistema, luego la pregunta del usuario
    # SystemMessage → instrucciones permanentes para el modelo (quién es, qué debe hacer)
    # HumanMessage  → la pregunta real del usuario
    messages = [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=ROUTER_HUMAN.format(query=query)),
    ]

    # Enviamos los mensajes a la API de OpenAI y esperamos la respuesta
    response = llm.invoke(messages)

    # Intentamos parsear la respuesta como JSON
    # El prompt instruye al modelo a responder SÓLO con: {"agent": "...", "reason": "..."}
    try:
        result = json.loads(response.content)
        agent = result.get("agent", "researcher")

        # Validamos que el agente elegido sea uno de los tres válidos
        # Si el modelo alucina y pone un nombre raro, lo corregimos a "researcher"
        if agent not in ("researcher", "code", "rag"):
            agent = "researcher"

        return {"agent": agent, "reason": result.get("reason", "")}

    except (json.JSONDecodeError, KeyError):
        # Si el JSON viene malformado (raro, pero posible), usamos researcher como fallback seguro
        return {"agent": "researcher", "reason": "fallback - parse error"}
