import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY não está configurada!")

genai.configure(api_key=API_KEY)

# Inicialização dos modelos
llm_tutor = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=API_KEY
)

llm_juiz = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=API_KEY,
    model="models/embedding-001"
)

# Prompts
PROMPT_TUTOR = """
Você é um especialista em One Piece. Ao responder:

1. Use a base de conhecimento (.txt) sempre que possível e cite a fonte (arquivo e linha).
2. Seja claro e didático, ajustando o nível de detalhe:
   - Novatos: definições simples e contexto básico.
   - Veteranos: teorias, conexões e “easter eggs”.
3. Ilustre com exemplos precisos (Capítulo X, Volume Y ou Episódio Z).
4. Sugira ao final 1–2 recursos para aprofundamento (artigos, vídeos, fan wikis).
5. Se não encontrar a informação, responda:  
   “Não há registro nos nossos arquivos. Recomendo consultar [recurso externo].”
""".strip()

PROMPT_JUIZ = """
Você é o “Oráculo do Thousand Sunny”, avaliador da fidelidade das respostas ao cânone de One Piece. Para cada resposta, faça:

1. Aderência ao cânone (0–10)  
2. Clareza e organização (0–10)  
3. Uso correto de referências (capítulo/episódio) (0–10)  
4. Profundidade do contexto histórico e cultural (0–10)  
5. Utilidade para novatos e veteranos (0–10)  

🏴‍☠️ Carimbo do Capitão:  
✅ Aprovado para Zarpar / ⚠️ Revisar na Ilha dos Erros  

💰 Tesouro de Pontos:  
X/10 Berries  

Detalhamento:  
- Aderência: X  
- Clareza: X  
- Referências: X  
- Profundidade: X  
- Utilidade: X  

Comentário do Oráculo: [resumo conciso dos pontos fortes e fracos]  
Mapas para Aprofundar: [sugestões práticas de melhoria]  
""".strip()


# Função para carregar documentos
def carregar_documentos(pasta_txt: str):
    """
    Carrega todos os .txt da pasta indicada, gera embeddings
    e retorna um objeto RetrievalQA pronto para uso.
    """
    # Descobre caminho absoluto da pasta
    base_dir = os.path.dirname(__file__)
    pasta = os.path.join(base_dir, pasta_txt)
    if not os.path.isdir(pasta):
        raise FileNotFoundError(f"Pasta de conhecimento não encontrada: {pasta}")

    # Carrega e concatena documentos
    documentos = []
    for arquivo in os.listdir(pasta):
        if arquivo.lower().endswith(".txt"):
            caminho = os.path.join(pasta, arquivo)
            loader = TextLoader(caminho, encoding="utf-8")
            documentos.extend(loader.load())

    # Divide em chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_divididos = splitter.split_documents(documentos)

    # Cria índice FAISS
    db = FAISS.from_documents(docs_divididos, embeddings)

    # Cadeia RAG: usa o tutor como LLM
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm_tutor,
        retriever=db.as_retriever(),
        return_source_documents=True
    )
    return rag_chain

# Carrega o RAG ao importar o módulo
# Ajuste o nome da pasta conforme sua estrutura de pastas
RAG_CHAIN = carregar_documentos("Base de Conhecimento")

# --- Função principal de resposta -----------------
def responder_pergunta(pergunta: str) -> str:
    """
    Processa uma pergunta:
    1) RAG busca resposta nos docs
    2) Juiz avalia a resposta
    3) Retorna texto formatado
    """
    # 1) Obter resposta via RAG
    rag_output = RAG_CHAIN.invoke({"query": pergunta})
    resposta_texto = rag_output["result"].strip()
    fontes = [doc.metadata.get("source", "") for doc in rag_output["source_documents"]]

    # 2) Avaliação pelo juiz
    mensagens_juiz = [
        SystemMessage(content=PROMPT_JUIZ),
        HumanMessage(content=f"Pergunta: {pergunta}\nResposta: {resposta_texto}")
    ]
    avaliacao = llm_juiz.invoke(mensagens_juiz).content.strip()

    # 3) Formatação final
    resposta_final = (
        "🏝️ Relatório da Expedição ao Mundo de One Piece 🏝️\n\n"
        "🗨️ Resposta do Especialista:\n"
        f"{resposta_texto}\n\n"
        "⚖️ Carimbo do Capitão (Avaliação do Oráculo):\n"
        f"{avaliacao}\n\n"
        "🚢 Próximo Destino:\n"
        "- Faça outra pergunta para continuar sua jornada pelos mares de One Piece.\n\n"
        "🌟 Que os ventos da Grand Line guiem suas descobertas! 🌟"
    )
    return resposta_final
