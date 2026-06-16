# scripts/evaluate_triad.py
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv # <-- 1. Añadimos esta importación

# <-- 2. Cargamos las variables del .env ANTES de hacer nada más
load_dotenv() 

# Añadimos la raíz del proyecto para poder importar tus módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.graph_rag import GraphRAG
from agents.rag_agent import run_rag_agent

# ─── 1. Definimos la estructura de la puntuación ───────────────────────
class TriadScore(BaseModel):
    score: int = Field(description="Puntuación del 1 al 5")
    reasoning: str = Field(description="Justificación detallada de la nota")

def get_judge():
    # El juez debe ser determinista (temperature=0) y estructurado
    return ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(TriadScore)

# ─── 2. Funciones de Evaluación (La Tríada) ────────────────────────────

def evaluate_faithfulness(context: str, response: str) -> TriadScore:
    """Evalúa si la respuesta inventa datos que NO están en el contexto."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un evaluador experto. Comprueba si la RESPUESTA está basada ÚNICAMENTE en el CONTEXTO. Puntúa de 1 (alucinación total) a 5 (completamente fiel)."),
        ("human", "CONTEXTO:\n{context}\n\nRESPUESTA:\n{response}")
    ])
    return (prompt | get_judge()).invoke({"context": context, "response": response})

def evaluate_answer_relevance(query: str, response: str) -> TriadScore:
    """Evalúa si la respuesta soluciona realmente la pregunta original."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un evaluador experto. Puntúa de 1 a 5 si la RESPUESTA aborda directamente y de forma útil la PREGUNTA del usuario."),
        ("human", "PREGUNTA:\n{query}\n\nRESPUESTA:\n{response}")
    ])
    return (prompt | get_judge()).invoke({"query": query, "response": response})

def evaluate_context_relevance(query: str, context: str) -> TriadScore:
    """Evalúa si GraphRAG trajo información útil."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un evaluador experto. Puntúa de 1 a 5 si la información del CONTEXTO contiene la respuesta necesaria para la PREGUNTA."),
        ("human", "PREGUNTA:\n{query}\n\nCONTEXTO:\n{context}")
    ])
    return (prompt | get_judge()).invoke({"query": query, "context": context})

# ─── 3. Ejecución contra tu sistema real ───────────────────────────────
if __name__ == "__main__":
    pregunta_prueba = "¿Qué diferencia hay entre RAG y GraphRAG?"
    print(f"🧐 Evaluando pregunta: '{pregunta_prueba}'\n")

    # A. Extraemos el contexto usando tu propia clase GraphRAG
    graph_rag = GraphRAG()
    contexto = graph_rag.retrieve(pregunta_prueba, k=3)
    
    # B. Generamos la respuesta llamando a tu agente
    respuesta = run_rag_agent(pregunta_prueba)

    # C. Pasamos los datos al juez
    nota_fidelidad = evaluate_faithfulness(contexto, respuesta)
    nota_resp = evaluate_answer_relevance(pregunta_prueba, respuesta)
    nota_ctx = evaluate_context_relevance(pregunta_prueba, contexto)

    print("📊 RESULTADOS DE LA TRÍADA RAG:")
    print(f"1. Precisión del Contexto: {nota_ctx.score}/5 -> {nota_ctx.reasoning}")
    print(f"2. Fidelidad (Faithfulness): {nota_fidelidad.score}/5 -> {nota_fidelidad.reasoning}")
    print(f"3. Relevancia Respuesta: {nota_resp.score}/5 -> {nota_resp.reasoning}")