# Desafio MBA Engenharia de Software com IA - Full Cycle

Este projeto implementa um sistema de RAG (Retrieval-Augmented Generation) para ingestão de PDFs e busca semântica utilizando PostgreSQL com pgVector, LangChain e modelos de embedding (OpenAI ou Google Gemini).

## Pré-requisitos

Certifique-se de ter instalado em sua máquina:

- [Python 3.10+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/) e [Docker Compose](https://docs.docker.com/compose/)

## Configuração

1. **Clone o repositório** (se ainda não o fez):
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd mba-ia-desafio-ingestao-busca
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure as variáveis de ambiente:**
   Copie o arquivo de exemplo `.env.example` para `.env`:
   ```bash
   cp .env.example .env
   ```
   Edite o arquivo `.env` e preencha as chaves de API necessárias (`OPENAI_API_KEY` ou `GOOGLE_API_KEY`). Você pode ajustar outras configurações como o caminho do PDF (`PDF_PATH`) se desejar.

## Banco de Dados

Suba o contêiner do PostgreSQL com pgVector utilizando o Docker Compose:

```bash
docker-compose up -d
```

Aguarde alguns instantes até que o banco de dados esteja pronto e a extensão `vector` seja ativada (o serviço `bootstrap_vector_ext` cuida disso).

## Execução

### 1. Ingestão de Documentos

Para processar o arquivo PDF (padrão: `document.pdf`) e armazenar os vetores no banco de dados:

```bash
python src/ingest.py
```

### 2. Chat Interativo

Para iniciar o chat e fazer perguntas sobre o conteúdo do PDF:

```bash
python src/chat.py
```

### 3. Busca Simples (Debug)

Para realizar uma busca semântica simples e ver os chunks retornados com seus scores:

```bash
python src/search.py "Sua pergunta aqui"
```