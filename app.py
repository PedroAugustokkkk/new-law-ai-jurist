# -*- coding: utf-8 -*-

"""
Serviço de RAG (Retrieval-Augmented Generation) para o "Jurist-AI".

v1.4: Corrige bug de 'UnicodeDecodeError' no BSHTMLLoader.
Removida a suposição de 'utf-8' no 'open_encoding', permitindo que
o BeautifulSoup auto-detecte o charset do arquivo HTML salvo.
"""

# --- Importações ---
import streamlit as st
import os
import re
from dotenv import load_dotenv

# LLM e Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain Core (Prompts)
from langchain_core.prompts import ChatPromptTemplate

# Imports de Chain (Classic)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# LangChain I/O e Armazenamento
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyPDFLoader, 
    TextLoader,
    BSHTMLLoader # O LEITOR DE HTML
)

# Processamento de Texto
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configurações Globais ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
STATIC_DIR = "static" 
STOPWORDS = set(['o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 'com', 'para', 'em', 
                 'um', 'uma', 'como', 'e', 'que', 'qual', 'eu', 'meu', 'minha', 'ver', 
                 'fazer', 'gerar', 'onde', 'está', 'vejo', 'posso', 'como', 'funciona',
                 'artigo', 'lei', 'paragrafo', 'inciso', 'caput'])

# --- 3. Indexação e Cache (ATUALIZADO) ---

@st.cache_resource 
def load_and_index_documents():
    """
    Carrega e indexa documentos de /data (PDFs, TXTs e HTMLs).
    
    Utiliza HuggingFaceEmbeddings ('all-MiniLM-L6-v2') para rodar a
    indexação localmente.
    """
    documents = []
    try:
        # Carregador de PDFs
        pdf_loader = DirectoryLoader("./data", glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
        
        # Carregador de TXTs
        txt_loader = DirectoryLoader("./data", glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        documents.extend(txt_loader.load())

        # --- CORREÇÃO (v1.4) ---
        # Removemos o 'loader_kwargs={"open_encoding": "utf-8"}'
        # para permitir a auto-detecção de encoding pelo BeautifulSoup.
        html_loader = DirectoryLoader(
            "./data", 
            glob="**/*.html", 
            loader_cls=BSHTMLLoader,
            show_progress=True
        )
        documents.extend(html_loader.load())
        # --- FIM DA CORREÇÃO ---

        if not documents:
            st.error("Diretório 'data' não encontrado ou vazio. Nenhum documento foi indexado.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        return vector_store.as_retriever(search_kwargs={"k": 5})
    
    except Exception as e:
        st.error(f"Erro during a indexação dos documentos: {e}")
        return None

# --- 4. Configuração da Chain de RAG ---
# (Esta função permanece inalterada)
def setup_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0
    )
    prompt_template = """
    Você é o "Jurist-AI", um assistente de pesquisa jurídica especialista em
    legislação e doutrina brasileira.
    
    Sua única função é responder perguntas de advogados e analistas, baseando-se
    *exclusivamente* nos textos de lei (ex: Emendas Constitucionais, Códigos) e
    artigos de doutrina fornecidos no contexto abaixo.

    Sua linguagem deve ser técnica, precisa e direta.
    Sempre que possível, cite o artigo ou a fonte da sua resposta.

    *** REGRA DE OURO (NÃO-ALUCINAÇÃO) ***
    NÃO invente informações, artigos ou interpretações.
    Se a resposta não estiver explicitamente nos textos fornecidos,
    sua *unica* resposta deve ser:
    "Não encontrei uma resposta direta para esta consulta nos textos indexados."

    Contexto (Textos de Lei e Doutrina):
    {context}
    
    Pergunta do Advogado:
    {input}
    
    Resposta Técnica (baseada nos textos):
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# --- 5. Lógica de Match de Imagem ---
# (Esta função permanece inalterada)
def clean_text_to_keywords(text: str) -> set:
    text = text.lower()
    text = re.sub(r'[\._-]', ' ', text)
    text = re.sub(r'\.(png|jpg|jpeg|gif)$', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    keywords = set(
        word for word in text.split() 
        if word not in STOPWORDS and len(word) > 2
    )
    return keywords

@st.cache_data
def get_image_keywords_map() -> dict:
    if not os.path.exists(STATIC_DIR):
        return {}
    
    image_map = {}
    for filename in os.listdir(STATIC_DIR):
        if filename.startswith('.'):
            continue
        keywords = clean_text_to_keywords(filename)
        if keywords:
            image_map[filename] = keywords
    return image_map

def find_best_image_match(query: str, image_map: dict) -> str | None:
    query_keywords = clean_text_to_keywords(query)
    if not query_keywords:
        return None
    best_match_file = None
    max_score = 0
    for filename, filename_keywords in image_map.items():
        score = len(query_keywords.intersection(filename_keywords))
        if score > max_score:
            max_score = score
            best_match_file = filename
    if max_score >= 2:
        return os.path.join(STATIC_DIR, best_match_file)
    else:
        return None

# --- 6. Interface Streamlit (UI) ---
# (Esta função permanece inalterada)
def main():
    st.set_page_config(page_title="Jurist-AI", page_icon="⚖️")
    st.title("⚖️ Jurist-AI (Protótipo Jusbrasil)")
    st.write("Assistente de pesquisa de legislação (gemini-2.5-flash)")

    try:
        retriever = load_and_index_documents()
        if not retriever:
            st.error("Falha ao carregar o índice de legislação. Verifique a pasta 'data'.")
            return 

        image_map = get_image_keywords_map()
        rag_chain = setup_rag_chain(retriever)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "image_path" in message:
                    st.image(message["image_path"])

        if user_query := st.chat_input("Ex: Como o 'imposto seletivo' afeta a Zona Franca?"):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.spinner("Consultando legislação e doutrina..."):
                response = rag_chain.invoke({"input": user_query})
                ai_response = response["answer"]
                image_path = find_best_image_match(user_query, image_map)

            with st.chat_message("assistant"):
                st.markdown(ai_response)
                if image_path:
                    st.image(image_path, caption="Fluxograma Relevante")
            
            message_data = {"role": "assistant", "content": ai_response}
            if image_path:
                message_data["image_path"] = image_path
            st.session_state.messages.append(message_data)

    except Exception as e:
        st.error(f"Ocorreu um erro crítico na aplicação: {e}")
        st.exception(e) 

# --- Ponto de Entrada ---
if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        st.error("Chave GOOGLE_API_KEY não encontrada nos segredos ou .env.")
    else:
        if not os.path.exists(STATIC_DIR):
            os.makedirs(STATIC_DIR)
        main()