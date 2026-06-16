# ─── Agente Researcher: responde preguntas conceptuales sobre IA ──────────────
# Este agente se activa cuando la pregunta es teórica o explicativa:
# "¿Qué es X?", "¿Por qué se usa Y?", "Compara A con B", etc.
# No necesita buscar en la base de datos: usa el conocimiento general del modelo.

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

# Importamos los prompts específicos para este agente
from prompts.researcher_prompt import RESEARCHER_HUMAN, RESEARCHER_SYSTEM


# @traceable registra esta función en LangSmith como un span de tipo "llm"
# run_type="llm" indica que es una llamada directa a un modelo de lenguaje
@traceable(name="researcher_agent", run_type="llm")
def run_researcher(query: str) -> str:
    """Responde preguntas conceptuales usando el conocimiento general de gpt-4o-mini."""

    # temperature=0.3 → permite algo de creatividad para explicaciones más naturales,
    # pero sigue siendo bastante determinista (0 = robótico, 1 = muy creativo)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Montamos el contexto de la conversación:
    # - SystemMessage: define el rol del agente (experto en IA/ML)
    # - HumanMessage: la pregunta real del usuario
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM),
        HumanMessage(content=RESEARCHER_HUMAN.format(query=query)),
    ]

    # Llamamos a la API y devolvemos el texto de la respuesta directamente
    response = llm.invoke(messages)
    return response.content
