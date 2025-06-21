from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# --- MCP Definitions ---

@dataclass
class Message:
    sender: str
    intent: str
    payload: Dict[str, Any]

@dataclass
class ContextStore:
    """Shared context for agents: stores messages, predictions, and metadata."""
    messages: List[Message] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    def write_message(self, msg: Message):
        self.messages.append(msg)

    def write_annotation(self, annotation: Dict[str, Any]):
        self.annotations.append(annotation)

    def get_history(self) -> List[Message]:
        return self.messages

@dataclass
class Plan:
    """Defines the ordered steps of the agent workflow."""
    steps: List[str]
    current_step: int = 0

    def next_step(self) -> Optional[str]:
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            self.current_step += 1
            return step
        return None

# --- Agent Implementation ---

class LoginClassifierAgent:
    def __init__(self, 
                 vectorstore: FAISS, 
                 llm: ChatGoogleGenerativeAI,
                 context: ContextStore,
                 k: int = 5):
        self.vs = vectorstore
        self.llm = llm
        self.context = context
        self.plan = Plan(steps=[
            "retrieve_history",
            "classify_record",
            "finalize"
        ])
        self.k = k

    def handle(self, msg: Message) -> str:
        self.context.write_message(msg)
        output = None

        # Execute plan steps
        while (step := self.plan.next_step()):
            if step == "retrieve_history":
                output = self._retrieve_history(msg)
            elif step == "classify_record":
                output = self._classify_record(msg, output)
            elif step == "finalize":
                output = self._finalize(msg, output)
        return output

    def _retrieve_history(self, msg: Message) -> List[str]:
        record_text = msg.payload["record_text"]
        similar = self.vs.similarity_search(record_text, k=self.k)
        history = [doc.page_content for doc in similar]
        self.context.write_annotation({"history": history})
        return history

    def _classify_record(self, msg: Message, history: List[str]) -> Dict[str, Any]:
        record_text = msg.payload["record_text"]
        # build prompt
        examples = "\n\n".join(f"Example {i+1}: {h}" for i, h in enumerate(history))
        prompt = f"""
Below are {self.k} similar login records for context:
{examples}

Now, given the record below, answer ONLY \"Yes\" or \"No\" to whether this login was from a stolen device, and in one sentence explain why.

Record to classify: {record_text}
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        prediction = response.content.strip()
        annotation = {"prediction": prediction}
        self.context.write_annotation(annotation)
        return annotation

    def _finalize(self, msg: Message, annotation: Dict[str, Any]) -> str:
        # Here you could trigger downstream tasks, alerts, etc.
        result = annotation["prediction"]
        return result

# --- Setup and Usage ---

if __name__ == "__main__":
    # 1) Load & index all rows
    df = pd.read_csv("LoginData.csv")
    def row_to_text(r):
        return (
            f"TS: {r['Login Timestamp']} | UID: {r['User ID']} | "
            f"RTT={r['Round-Trip Time [ms]']}ms | IP={r['IP Address']} | "
            f"City={r['City']}, {r['Country']} | "
            f"Success={r['Login Successful']} | "
            f"AttackIP={r['Is Attack IP']} | ATO={r['Is Account Takeover']}"
        )
    texts = df.apply(row_to_text, axis=1).tolist()
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_texts(texts, emb, metadatas=[{"row": i} for i in df.index])

    # 2) Setup LLM and context
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    context = ContextStore()

    # 3) Instantiate agent
    agent = LoginClassifierAgent(vs, llm, context)

    # 4) Create an MCP message for your target record
    target_idx = 37
    record_text = texts[target_idx]
    msg = Message(
        sender="User",
        intent="classify_login",
        payload={"record_text": record_text}
    )

    # 5) Run the agent
    result = agent.handle(msg)
    print("Stolen device prediction:", result)
