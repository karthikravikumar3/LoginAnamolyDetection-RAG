
## 🧠 How It Works

1. **Login Record Processing**
   - Each login record includes metadata like timestamp, IP, location, RTT, user agent, login result, and threat indicators.
   - Records are preprocessed into descriptive natural language sentences for embedding.

2. **Embedding & Indexing**
   - Each login sentence is embedded using `all-MiniLM-L6-v2` from HuggingFace.
   - Embeddings are stored in a **FAISS** index for similarity retrieval.
   - This step is done once and reused — no need for runtime reprocessing.

3. **Risk Evaluation via LLM**
   - Given a new login attempt, the system retrieves the top-k most similar past logins.
   - Constructs a prompt combining historical context and the new record.
   - Sends the prompt to **Gemini LLM** via **LangChain** for classification.
   - LLM returns a risk probability and a one-sentence rationale.

---
## Sample prompt
  Now, given the records, rate the probability out of 100 to whether this login was from a stolen device, and in one sentence explain why.
  
  Record to classify: TS: 2020-02-03 12:46:09.512 | UID: -4324475583306591935 | RTT=50000ms | IP=162.202.10.55 | City=London, Country=GB | Success=True | AttackIP=True | ATO=False

## Sample Output
  Probability: 91/100  
  Reason: Login originated from a known attack IP and a new country (GB) not seen in prior successful logins, indicating high risk of compromise.

Datasets from Kaggle Dataset (https://www.kaggle.com/datasets/dasgroup/rba-dataset/data)
