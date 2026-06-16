# ─── Agente Code: genera y explica código Python ──────────────────────────────
# Este agente se activa cuando la pregunta pide código:
# "Escribe un ejemplo de...", "Cómo implemento...", "Muéstrame código para..."
# Está especializado en Python moderno con LangChain, LangGraph y OpenAI API.

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

# Importamos los prompts específicos para generación de código
from prompts.code_agent_prompt import CODE_AGENT_HUMAN, CODE_AGENT_SYSTEM


# @traceable registra esta función en LangSmith como un span de tipo "llm"
@traceable(name="code_agent", run_type="llm")
def run_code_agent(query: str) -> str:
    """Genera código Python respondiendo a la petición del usuario."""

    # temperature=0.1 → casi determinista
    # Para código es importante que sea correcto y reproducible, no creativo
    # Un valor más alto podría generar código con variaciones o errores aleatorios
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # Construimos la conversación con el rol de experto en Python/IA
    messages = [
        SystemMessage(content=CODE_AGENT_SYSTEM),
        HumanMessage(content=CODE_AGENT_HUMAN.format(query=query)),
    ]

    # Llamamos a la API y devolvemos el código generado como string
    response = llm.invoke(messages)
    return response.content
