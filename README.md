# 📧 Smart Email Assistant using ML, GenAI & Agentic AI

A multi-agent system that automates internal company email handling by classifying emails, generating responses using LLMs, and escalating ambiguous or low-confidence messages. Built using **CrewAI**, **Hugging Face LLMs**, and **ML classification models**.

---

## 🚀 Project Overview & Architecture

This project implements an **agentic framework** using [CrewAI](https://docs.crewai.com/) with the following agents:

```

                                 [ Email Input ]
                                        |
                         [ Email Classifier Agent (ML) ]
                                        |
        ┌───────────── Confidence < 0.6 or Category = 'Other' ─────────────┐
        ↓                                                                  ↓
[ Response Generator Agent (LLM) ]                             [ Escalation Agent ]
        ↓                                                                 ↓
[ Email Reply ]                                               [ Escalation Logged ]

```

### 🧠 Agents

| Agent                   | Function                                                |
|------------------------|---------------------------------------------------------|
| Email Classifier Agent | Uses a supervised ML model to classify emails.          |
| Response Generator     | Uses an open-source LLM to generate smart replies.      |
| Escalation Agent       | Flags low-confidence or uncertain category emails.      |

---

## 🧪 ML Model Details & Evaluation Metrics

### 📊 Classifier

- **Model**: Logistic Regression
- **Vectorizer**: TF-IDF
- **Dataset**: 300 synthetic emails across 3 categories: `HR`, `IT`, `Other`

### 📈 Evaluation

| Metric       | Value (approx) |
|--------------|----------------|
| Accuracy     | 0.95+          |
| Precision    | High across classes |
| Recall       | High across classes |

- **Confusion Matrix** and **Classification Report** are included in the training notebook.
- Threshold for confident classification: `0.6`

---

## 🧠 Prompt Design & LLM Integration

### 🔗 LLM Used

- Model: [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama)
- Pipeline: `transformers.pipeline("text-generation")`
- Provider: Hugging Face 🤗

### 📝 Prompt Templates

Prompts are dynamically adapted per category:

- **IT**:
  > You're an IT support assistant. Write a concise and professional reply to the following employee email, offering help or next steps.

- **HR**:
  > You're from the HR department. Write a formal and informative response to the employee's query mentioned below.

- **Other**:
  > You're a general assistant for company internal communications. Write a polite and helpful response to the following query.

---

## ⚙️ Setup & Execution Instructions

### 📁 Folder Structure

```

smart-email-assistant/
├── agents/
│   ├── email_classifier.py
│   ├── response_generator.py
│   └── escalation_agent.py
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
├── data/
│   └── emails.csv
├── notebooks/
│   └── train_email_classifier.ipynb
├── orchestrator.py
├── app.py
├── requirements.txt
└── README.md

````

---

### 🧑‍💻 Run Locally (Streamlit App)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run app.py
````

---

### 🧠 Train the Model (Optional)

```bash
# Open the notebook to train & evaluate the model
jupyter notebook notebooks/train_email_classifier.ipynb
```

This generates:

* `models/model.pkl`
* `models/vectorizer.pkl`

---

## ✅ Deliverables

* ✅ Modular agent implementation (CrewAI)
* ✅ Model training notebook & evaluation
* ✅ Inference-ready ML model
* ✅ Prompt-tuned LLM agent
* ✅ Streamlit interface for live testing


---

## 👨‍💻 Author

**Rohit Jakkam**
🔗 [LinkedIn](https://www.linkedin.com/in/rohitjakkam/)
🔗 [GitHub](https://github.com/Rohitjakkam)
