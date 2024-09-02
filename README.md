## PDF-Whisperer

This project develops an Intelligent PDF Reader using Large Language Models (LLMs) to read, understand, and interact with PDF documents. The system extracts relevant information, understands context and semantics, and enables users to ask questions, request summaries, or engage in discussions based on the PDF content. Advanced features include entity recognition, contextual understanding, sentiment analysis, and interactive querying. The project supports two modes of operation: 

(1) using an OpenAI API key for cloud-based processing and 

(2) running locally with Ollama and Litellm for offline processing.

The project can leverages pre-trained LLMs, fine-tuned llm , and ensemble methods to produce a comprehensive understanding of PDF content. Users can choose to use their OpenAI API key for cloud-based processing or run the system locally using Ollama and Litellm, providing flexibility and control over data processing. The goal is to automate information extraction, enhance user interaction, and support decision-making. Target audiences include researchers, professionals, and students seeking to simplify information extraction, enhance productivity, and facilitate learning. By harnessing the power of LLMs, this project revolutionizes the way we interact with PDF documents.


```bash
pip -r install requirements.txt 
```

### invoke api_key for using OpenAI's GPT model

![alt text](image-1.png)

###run this following command to start streamlit

```bash
streamlit run app.py
```

### if running using local llm 

![alt text](image.png)

[ollama]( https://ollama.com/download/)

[Quick Start - LiteLLM Proxy CLI](https://docs.litellm.ai/docs/proxy/quick_start#quick-start---litellm-proxy--configyaml)
Run the following command to start the litellm proxy

```bash
litellm --model ollama/<model_name>
```
Proxy running on http://0.0.0.0:4000

use this url had base_url in place of openai_api_key