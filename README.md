# 🏥 LLM Ensemble Hallucination Detector

## 🚀 Overview
LLM-Ensemble-Hallucination_Detector is a Python-based application designed to evaluate and mitigate hallucinations in Large Language Models (LLMs). It implements an ensemble approach by leveraging multiple LLM APIs to generate and assess responses, using sentence-level semantic similarity and reference-based judgment techniques. This tool aids in identifying inconsistencies or hallucinated outputs and supports comparative model evaluation.

## 📌 Features
🔗 **LLM Ensemble API Integration:** Queries multiple large language models (LLMs) like "llama3-70b-8192", "qwen-qwq-32b", "gemma2-9b-it", and "deepseek-r1-distill-llama-70b", from OpenRouter, Groq, and others via API to get responses.

🧠 **Semantic Summarization:** Accepts free-text input and generates a single-word distilled response from each model.

🌐 **Flask Backend:** Simple web API built using Flask to serve LLM responses via HTTP requests.

✅ **Fallback & Error Handling:** If one model fails, the app continues with available ones and logs failures gracefully.

🛡️ **Secure Key Handling:** Uses .env file to securely store API keys instead of hardcoding them in source files.

🔍 **Minimalistic Output:** Designed for quick interpretation — the output is concise and meant for fast comparison of model behavior.


## 🛠️ Technologies Used
- **Backend Language:** Python3
- **Web:** HTML, CSS, JS, Flask
- **APIs Used:** Groq API, OpenRouter API
- **ML/NLP:** Hugging Face sentence-transformers, Transformers
- **Dataset:** TruthfulQA
- **Version Control:** Git, Github
- **Development Tools:** VS Code


## 📌 Installation & Setup
### Clone the repository
```bash
git clone https://github.com/praneetware/LLM-Ensemble-Hallucination_Detector.git
```
### Navigate to project directory
```bash
cd LLM-Ensemble-Hallucination_Detector
```
### Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### Set up your environment variables
```env
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```
### Run the application
```bash
python app.py
```
 

## 🚀 Usage
-The app will query multiple LLMs for a given prompt or question.

-It will compare model outputs against a reference answer using semantic similarity.

-Scores and hallucination flags will be printed or logged.

-With help of NLP, the dataset answers will be compared with the LLM answer and output produced will be short and to-the-point.

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## 📄 License
This project is licensed under the **BSD 3-Clause License**.

---

 **Helping people find accurate results!**

