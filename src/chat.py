import os
import logging
import sys
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.search import search

# Configure logging to show less noise for chat
logging.getLogger("src.ingest").setLevel(logging.WARNING)
logging.getLogger("src.search").setLevel(logging.WARNING)

def get_llm():
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if openai_key:
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano")
        # Fallback if gpt-5-nano doesn't exist yet in the library or API
        # Using a widely available one if needed, or trusting the user input
        return ChatOpenAI(model=model, temperature=0)
    elif google_key:
        model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    else:
        raise ValueError("Neither OPENAI_API_KEY nor GOOGLE_API_KEY set.")

def chat_loop():
    print("Bem-vindo ao Chat do PDF! Digite 'sair' para encerrar.")
    
    llm = get_llm()
    
    prompt_template = """CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""
    prompt = PromptTemplate.from_template(prompt_template)

    while True:
        try:
            user_input = input("\nFaça sua pergunta: ").strip()
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Encerrando...")
                break
            
            if not user_input:
                continue

            # 1. Search
            results = search(user_input, k=10)
            
            # Concatenate context
            context_str = "\n\n".join([doc.page_content for doc, _ in results])
            
            # 2. Generate Answer
            chain = prompt | llm
            response = chain.invoke({"context": context_str, "question": user_input})
            
            print(f"\nRESPOSTA: {response.content}")
            
        except KeyboardInterrupt:
            print("\nEncerrando...")
            break
        except Exception as e:
            print(f"Erro: {e}")

if __name__ == "__main__":
    chat_loop()