# PDF-QAChatbot-with-Rag
[Link to PDF-QAChatbot-with-Rag Application](https://pdf-appchatbot-with-rag.streamlit.app/)

Step 1: Upload a PDF file in the left sidebar, then click 'Upload Documents' to start processing the document.

Step 2: Once the process finishes, start asking questions like summarization or explanation of specific knowledge based on the given document.
![](https://github.com/TeKaiChou/PDF-QAChatbot-with-Rag/blob/main/PDFQA_RAG.gif)


## Overview
This is a web application of QA Chatbot with RAG that combines Streamlit, LlamaIndex, and Gemini-Pro LLM, aiming to simplify PDF document analysis. With retrieval augmented generation (RAG), we can realize dynamic and interactive dialogue according to the context from provided documents and chat history, which is fit for efficient document retrieval and summarization. 

You can replace the LLM with other popular open-source models, connecting to API, or your model. For the vector database used for RAG, here I use the VectorStoreIndex (in-memory) from LlamaIndex as storage context. It's more suitable to have your own vector database locally or services(ex: Pinecone, Chroma, etc.) as a storage context when you have a business or company-level requirement. You can also customize the QA template and the refined template to meet different purposes. 

## Prerequisites and dependencies
- Python
- python-dotenv (for reading environment variable)
- Streamlit
- Framework: Llamaindex (can also use langchain to deeper customize your LLM application)
- Google API Key (LLM: Gemini-pro, Embedding Model: embedding-001)

  (You can run the script to see the available LLM and Embedding models from Google Generativeai)
```
import google.generativeai as genai

# List Gemini models
print('-----------generateContent models-----------')
for models in genai.list_models():
  if 'generateContent' in models.supported_generation_methods:
    print(models.name)

print('-----------embedContent models-----------')
for models in genai.list_models():
  if 'embedContent' in models.supported_generation_methods:
    print(models.name)
```

## Reference:
- [LlamaIndex : Create, Save & Load Indexes, Customize LLMs, Prompts & Embeddings](https://medium.com/@reddyyashu20/llamaindex-create-save-load-indexes-customize-llms-prompts-embeddings-abb581df6dac)
- [LlamaIndex: Chat Engine - Context Mode](https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_context.html)
- [Multi-Modal LLM using Google’s Gemini model for image understanding and build Retrieval Augmented Generation with LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/multi_modal/gemini.html)
