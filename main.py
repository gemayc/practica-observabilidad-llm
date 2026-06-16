# ─── main.py: interfaz de línea de comandos (CLI) del sistema multi-agente ────
# Este archivo es la forma de usar el sistema desde la terminal, sin interfaz web.
# Muestra un menú de bienvenida, acepta preguntas en bucle y muestra las respuestas
# con colores usando la librería Rich.
#
# Ejecución: uv run python main.py
# Para salir: escribe "salir", "exit" o "quit"

import os

from dotenv import load_dotenv

# Cargamos el archivo .env con las claves de API antes de importar cualquier otra cosa
# (LangChain lee las variables de entorno al importarse, no al usarse)
load_dotenv()

# Rich: librería para texto con colores, tablas y paneles en la terminal
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# Importamos la función que construye y compila el grafo LangGraph
from graph.workflow import build_workflow

# Objeto Console de Rich: es el equivalente a print() pero con colores y estilos
console = Console()

# Texto de bienvenida en formato Markdown — Rich lo renderiza con formato
WELCOME_MD = """\
# 🤖 Multi-Agent AI Assistant
### Observabilidad LLM con LangSmith + LangGraph

**Agentes disponibles (seleccionados automáticamente por el router):**

| Agente | Cuándo se usa |
|--------|--------------|
| 🔬 Researcher | Preguntas conceptuales sobre IA/ML |
| 💻 Code | Generación y explicación de código Python |
| 🕸️  GraphRAG | Base de conocimiento: LangSmith, LangChain, RAG, métricas… |

Todos los pasos quedan trazados en **LangSmith** automáticamente.

Escribe `salir` para terminar.
"""

# Mapa de estilos por tipo de agente: color de Rich y etiqueta a mostrar
AGENT_STYLES = {
    "researcher": ("cyan",    "🔬 Researcher"),
    "code":       ("green",   "💻 Code Agent"),
    "rag":        ("magenta", "🕸️  GraphRAG Agent"),
}


def main() -> None:
    """Bucle principal de la CLI: muestra bienvenida, acepta preguntas y muestra respuestas."""

    # Leemos la configuración de LangSmith desde las variables de entorno
    langsmith_key     = os.getenv("LANGCHAIN_API_KEY", "")
    langsmith_project = os.getenv("LANGCHAIN_PROJECT", "practica-observabilidad-llm")
    tracing_on        = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    # Mostramos el panel de bienvenida con el texto Markdown
    console.print(Panel(Markdown(WELCOME_MD), border_style="blue"))

    # Mostramos el estado de LangSmith (activo o no)
    status_parts = [f"Proyecto: [bold]{langsmith_project}[/bold]"]
    if tracing_on and langsmith_key:
        status_parts.append("[green]LangSmith tracing: ON ✓[/green]")
    else:
        status_parts.append("[yellow]LangSmith tracing: OFF (configura LANGCHAIN_API_KEY)[/yellow]")
    console.print("  " + "  |  ".join(status_parts) + "\n")

    # Construimos y compilamos el grafo LangGraph (sólo se hace una vez al inicio)
    app = build_workflow()

    # Bucle principal: pedimos preguntas hasta que el usuario quiera salir
    while True:
        # Prompt.ask() muestra "> Pregunta:" en verde y espera que el usuario escriba
        query = Prompt.ask("[bold green]Pregunta[/bold green]").strip()

        # Ignoramos entradas vacías (el usuario pulsó Enter sin escribir nada)
        if not query:
            continue

        # Comprobamos si el usuario quiere salir
        if query.lower() in ("salir", "exit", "quit"):
            console.print("[yellow]¡Hasta luego![/yellow]")
            break

        # Ejecutamos el grafo LangGraph con la pregunta del usuario
        # El estado inicial tiene la pregunta; los demás campos los rellenan los nodos
        with console.status("[bold yellow]Pensando...[/bold yellow]", spinner="dots"):
            result = app.invoke(
                {"query": query, "agent_type": "", "agent_reason": "", "response": ""}
            )

        # Extraemos el agente que respondió y buscamos su color/etiqueta
        agent = result["agent_type"]
        color, label = AGENT_STYLES.get(agent, ("white", agent))

        # Mostramos qué agente respondió y la razón del router
        console.print(
            f"\n[{color}]{label}[/{color}]"
            + (f"  — _{result['agent_reason']}_" if result.get("agent_reason") else "")
        )

        # Mostramos la respuesta dentro de un panel con el color del agente
        # Markdown() renderiza el texto con negritas, listas, código, etc.
        console.print(Panel(Markdown(result["response"]), border_style=color))


# Solo ejecutamos main() si este archivo se llama directamente (no al importarlo)
if __name__ == "__main__":
    main()
