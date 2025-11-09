# ⚖️ Jurist-AI (Protótipo de RAG Jurídico)

> Um chatbot de RAG (Retrieval-Augmented Generation) que transforma nova legislação complexa (como a Reforma Tributária) em um assistente de pesquisa interativo.

Este protótipo (direcionado à Jusbrasil) demonstra como a IA pode resolver a maior dor do mundo jurídico: a **assimilação de novas leis**. O sistema ataca diretamente o problema de "excesso de informação" que define o core business da Jusbrasil.
Caso deseje testar agora, pode acessar a URL: https://ai-jurist.streamlit.app

## 🎯 O Problema

Quando uma lei complexa é aprovada (ex: Reforma Tributária, Marco Civil da IA), advogados, contadores e empresas levam meses para entender o impacto. Eles enfrentam milhares de páginas de texto de lei denso, artigos de doutrina e notícias. A pesquisa é manual, cara e lenta.

## 💡 A Solução

Um "Assistente Jurídico" (Jurist-AI) que usa uma arquitetura RAG para ler, indexar e "entender" essa nova base de conhecimento.

O sistema é alimentado com os PDFs da Emenda Constitucional (a lei) e artigos de análise (a doutrina). Um advogado pode então perguntar em linguagem natural: "Como o 'imposto seletivo' afeta empresas do Simples Nacional?" e receber uma resposta técnica, precisa e instantânea, baseada *exclusivamente* nos textos fornecidos.

**Valor para o Negócio (Jusbrasil):**
* **Time-to-Market:** Esta é uma *feature* que a Jusbrasil pode vender. Em vez de esperar 6 meses para analistas criarem conteúdo sobre a nova lei, a IA pode disponibilizar a consulta em 6 horas.
* **Privacidade (Diferencial):** Ao usar Embeddings Locais (HuggingFace), o protótipo garante que os dados (que podem ser documentos legais sensíveis) *nunca* saiam do servidor para serem indexados por uma API de terceiros.
* **Precisão (Guardrails):** O prompt da IA é configurado para `temperature=0.0` e instruído a *nunca* alucinar, respondendo "Não encontrei" se a informação não estiver no texto—uma regra de segurança crítica para LawTech.

## ✨ Funcionalidades Principais

* **RAG sobre Legislação:** Indexa múltiplos PDFs (leis, artigos) da pasta `/data`.
* **Prompt Jurídico (Guardrail):** O `prompt_template` é desenhado para ser um assistente técnico, preciso e que se recusa a "opinar" ou "alucinar", mantendo-se 100% aterrado ao contexto.
* **Embeddings Locais (Privacidade/Custo):** Utiliza `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) para vetorizar os documentos localmente, garantindo privacidade e custo zero de indexação.
* **Geração Rápida:** Utiliza o `gemini-2.5-flash` para respostas de baixa latência.

## 🛠️ Stack de Tecnologia

* **Frontend:** Streamlit
* **Orquestração RAG:** LangChain
* **LLM (Geração):** Google Gemini 2.5 Flash (via API)
* **Leitor de Documentos:** `PyPDFLoader` (via LangChain)
* **Embeddings (Vetorização):** Hugging Face `all-MiniLM-L6-v2` (Local)
* **Vector Store (Busca):** FAISS-CPU (em memória)

## 🚀 Como Executar Localmente

1.  Clone o repositório.
2.  Crie e ative um ambiente virtual.
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4.  Popule a base de conhecimento:
    * Adicione os PDFs da legislação (ex: EC 132) e artigos de análise na pasta `/data`.

5.  Configure suas chaves de API:
    * Crie um arquivo `.env` e adicione sua `GOOGLE_API_KEY`.

6.  Execute a aplicação:
    ```bash
    streamlit run app.py
    ```
