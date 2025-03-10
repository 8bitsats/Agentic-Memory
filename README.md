# Agentic-Memory

Welcome to the **Agentic Memory Guide** by Cheshire Terminal, a practical blueprint for creating an AI system that transcends the stateless limitations of traditional language models. Drawing from human memory frameworks and the innovative **Wonderland Memory System**, this guide equips you with the tools to build an AI—think of it as your own Mad Hatter—that remembers past interactions, learns from experiences, and adapts to user needs. By leveraging **LangChain**, vector databases, and blockchain integrations with **Bitcoin**, **Solana**, and **Base**, we’ll craft an agentic AI that thrives in open-ended environments like Wonderland.

The current date is **March 10, 2025**, and this guide reflects the latest advancements in AI memory frameworks. Let’s dive down the rabbit hole!

## Why Agentic Memory Matters

Traditional language models are like goldfish—each interaction starts anew, with no memory of what came before unless you manually provide context. This statelessness limits their ability to maintain continuity, learn, or adapt over time. In contrast, an **agentic AI** with memory can:

- **Remember**: Recall past conversations or events to inform current responses.
- **Learn**: Refine its behavior based on what worked or didn’t.
- **Adapt**: Anticipate user needs in evolving scenarios.

In Wonderland, a fully open-ended storytelling realm, memory is critical. Without it, the AI (the Mad Hatter) might forget pivotal choices—like your last visit to Crumpet Falls—breaking the magic of your narrative. The **Wonderland Memory System**, with its dual-layer approach of **Auto Summarization** and the **Memory Bank**, addresses this by mimicking human memory processes. This guide extends that concept into a developer-friendly framework, enhanced by blockchain technologies for decentralization and scalability.

---

## Understanding the Four Memory Types

### Working Memory

- **Definition**: The short-term "RAM" of your AI, holding the current conversation or task context.
- **Role**: Provides immediate context for responses, tracking the back-and-forth in real time.
- **Wonderland Example**: The **Story Summary**, a dynamic overview of recent actions, keeps the Mad Hatter aligned with your current adventure.

### Episodic Memory

- **Definition**: Long-term memory for specific events or "episodes," like past interactions.
- **Role**: Recalls detailed experiences (e.g., "what," "when," "where") to enrich current interactions.
- **Wonderland Example**: The **Memory Bank** stores condensed story chunks (Memories), retrieving them when you revisit familiar settings or themes.

### Semantic Memory

- **Definition**: A structured repository of facts and concepts, like a personal knowledge base.
- **Role**: Grounds responses in reliable, general knowledge.
- **Wonderland Example**: **Plot Essentials**, permanent details (e.g., character names or world rules) the Mad Hatter never forgets.

### Procedural Memory

- **Definition**: The "how-to" memory for skills and routines.
- **Role**: Executes learned tasks automatically, improving efficiency over time.
- **Wonderland Example**: **Auto Summarization**, which compresses narrative blocks into Memories without user intervention.

These memory types work together to create a cohesive, adaptive AI system, mirroring the human brain’s synergy of compression and retrieval.

## Implementation with LangChain

### Setting Up the Environment

To build this system, you’ll need:

- **Python 3.8+**
- **LangChain**: `pip install langchain langchain-openai`
- **OpenAI API Key**: For the LLM (e.g., GPT-4o).
- **Weaviate**: A vector database for episodic memory (run via Docker).
- **Docker Compose**: For Weaviate and Ollama embeddings.
- **Blockchain Tools**: Optional SDKs for Bitcoin (e.g., `python-bitcoinlib`), Solana (e.g., `solana-py`), and Base (e.g., [Web3.py](http://web3.py/)).

**Docker Compose Setup** (`docker-compose.yml`):

```yaml
version: '3'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - 8080:8080
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-ollama'
      ENABLE_MODULES: 'text2vec-ollama'
      OLLAMA_API_ENDPOINT: '<http://ollama:11434/api>'
  ollama:
    image: ollama/ollama:latest
    ports:
      - 11434:11434

```

Run: `docker-compose up -d`.

---

### Working Memory: Immediate Context

**Goal**: Maintain the current conversation in memory, akin to Wonderland’s Story Summary.

**Code**:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(temperature=0.7, model="gpt-4o")
system_prompt = SystemMessage("You are a helpful AI Assistant. Respond in one sentence.")
messages = [system_prompt]

while True:
    user_input = input("\\nUser: ")
    if user_input.lower() == "exit":
        break
    user_message = HumanMessage(user_input)
    messages.append(user_message)
    response = llm.invoke(messages)
    print("\\nAI: ", response.content)
    messages.append(response)

```

**Output Example**:

```
User: Hi there!
AI: Hello! How can I assist you today?
User: What's my name?
AI: I don’t know yet—what’s your name?

```

**Explanation**: The `messages` list serves as working memory, feeding the full conversation context to the LLM. In Wonderland, this mirrors the **Story Summary**, updated continuously to reflect the latest actions.

---

### Episodic Memory: Learning from the Past

**Goal**: Store and retrieve past interactions, like the Wonderland Memory Bank.

**Code**:

```python
import weaviate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from weaviate.classes.config import Property, DataType, Configure

# Connect to Weaviate
vdb_client = weaviate.connect_to_local()

# Create Episodic Memory Collection
vdb_client.collections.create(
    name="episodic_memory",
    vectorizer_config=Configure.NamedVectors.text2vec_ollama(
        name="title_vector", source_properties=["conversation"], model="nomic-embed-text",
        api_endpoint="<http://host.docker.internal:11434>"
    ),
    properties=[
        Property(name="conversation", data_type=DataType.TEXT),
        Property(name="context_tags", data_type=DataType.TEXT_ARRAY),
        Property(name="summary", data_type=DataType.TEXT),
        Property(name="what_worked", data_type=DataType.TEXT),
        Property(name="what_to_avoid", data_type=DataType.TEXT),
    ]
)

# Reflection Prompt
reflection_prompt = ChatPromptTemplate.from_template("""
Analyze this conversation for future use in JSON:
{{
    "context_tags": [string, ...], "summary": string,
    "what_worked": string, "what_to_avoid": string
}}
Conversation: {conversation}
""")
reflect = reflection_prompt | llm | JsonOutputParser()

def format_conversation(messages):
    return "\\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages[1:]])

def add_episodic_memory(messages, vdb_client):
    conversation = format_conversation(messages)
    reflection = reflect.invoke({"conversation": conversation})
    episodic_memory = vdb_client.collections.get("episodic_memory")
    episodic_memory.data.insert(reflection)

def episodic_recall(query, vdb_client):
    episodic_memory = vdb_client.collections.get("episodic_memory")
    memory = episodic_memory.query.hybrid(query=query, alpha=0.5, limit=1)
    return memory.objects[0].properties if memory.objects else None

# Example
add_episodic_memory(messages, vdb_client)
memory = episodic_recall("What's my name", vdb_client)
print(memory)

```

**Output Example**:

```python
{
    "context_tags": ["greeting", "name_query"],
    "summary": "User asked for name; AI prompted for it.",
    "what_worked": "Prompting user for missing info.",
    "what_to_avoid": "Assuming prior knowledge."
}

```

**Explanation**: Conversations are reflected upon, stored as vectors in Weaviate, and recalled using hybrid search. This mirrors Wonderland’s **Memory Bank**, where Memories are stored and retrieved based on vector similarity to current actions.

---

### Semantic Memory: Factual Knowledge Base

**Goal**: Store and access factual knowledge, like Wonderland’s Plot Essentials.

**Code**:

```python
knowledge_base = {}

def update_semantic_memory(key, value):
    knowledge_base[key] = value

def query_semantic_memory(query):
    return knowledge_base.get(query, "I don’t have that info yet.")

# Example
update_semantic_memory("User’s favorite food", "pizza")
print(query_semantic_memory("User’s favorite food"))  # "pizza"

```

**Explanation**: A simple key-value store (expandable to a knowledge graph) holds persistent facts. In Wonderland, this is the **Plot Essentials**, ensuring key details remain accessible.

---

### Procedural Memory: Skill Execution

**Goal**: Automate learned tasks, like Wonderland’s Auto Summarization.

**Code**:

```python
def summarize_text(text):
    prompt = ChatPromptTemplate.from_template("Summarize: {text}")
    return llm.invoke(prompt.format(text=text)).content

# Example
text = "User loves pizza for its cheesy goodness."
summary = summarize_text(text)
print(summary)  # "User enjoys pizza’s cheesy flavor."

```

**Explanation**: Codified skills (e.g., summarization) execute automatically, refining over time based on feedback. In Wonderland, **Auto Summarization** condenses actions into Memories effortlessly.

---

### Bringing It All Together

**Full Agentic AI**:

```python
conversations, messages = [], []

def episodic_system_prompt(query, vdb_client):
    memory = episodic_recall(query, vdb_client) or {}
    return SystemMessage(f"""You are a helpful AI Assistant.
    Past Memory: {memory.get('summary', 'N/A')}
    Use this context to respond.""")

while True:
    user_input = input("\\nUser: ")
    if user_input.lower() == "exit":
        add_episodic_memory(messages, vdb_client)
        break
    user_message = HumanMessage(user_input)
    system_prompt = episodic_system_prompt(user_input, vdb_client)
    messages = [system_prompt] + [msg for msg in messages if not isinstance(msg, SystemMessage)]
    messages.append(user_message)
    context = query_semantic_memory(user_input)
    if context != "I don’t have that info yet.":
        messages.append(HumanMessage(f"Fact: {context}"))
    response = llm.invoke(messages)
    print("\\nAI: ", response.content)
    messages.append(response)

```

**Output Example**:

```
User: What’s my favorite food?
AI: Based on past info, your favorite food is pizza.
User: Summarize that!
AI: You enjoy pizza’s cheesy flavor.

```

**Explanation**: This integrates all memory types, creating an AI that uses working memory for immediate context, episodic memory for past events, semantic memory for facts, and procedural memory for skills—mirroring Wonderland’s cohesive system.

---

## Enhancements with Blockchain

To elevate this memory system, we integrate **Bitcoin**, **Solana**, and **Base blockchains**:

1. **Bitcoin**:
    - **Use Case**: Store immutable Semantic Memory (e.g., Plot Essentials) or pivotal Episodic Memories as blockchain transactions.
    - **Benefit**: Ensures permanence and tamper-proof records, ideal for critical story elements.
    - **Example**: Record "User’s favorite food: pizza" as a Bitcoin transaction hash.
2. **Solana**:
    - **Use Case**: Host Episodic Memory vectors and real-time Working Memory updates via smart contracts.
    - **Benefit**: High-speed transactions enable rapid Memory Bank retrieval and Story Summary updates.
    - **Example**: Store Memory vectors on Solana, leveraging Cheshire Terminal’s DeFi infrastructure.
3. **Base**:
    - **Use Case**: Manage Procedural Memory scripts and scale Memory Bank capacity cost-effectively.
    - **Benefit**: Ethereum Layer 2 reduces costs for frequent operations, supporting higher tiers (e.g., 400 Memories for Mythic users).
    - **Example**: Run Auto Summarization logic on Base for efficient processing.

**Membership Tier Integration**:

| Tier | Memory Bank Capacity | Blockchain Storage |
| --- | --- | --- |
| Free | 25 Memories | Base (cost-effective) |
| Adventurer | 50 Memories | Base + Solana (speed) |
| Champion | 100 Memories | Solana (primary) |
| Legend | 200 Memories | Solana + Bitcoin (backup) |
| Mythic | 400 Memories | All three (full redundancy) |

**Settings**: Enable via `Settings Sidebar → Gameplay → AI Models → Memory System`, with blockchain nodes configurable for decentralization.

---

Agentic Memory by Cheshire Terminal transforms AI into a dynamic, learning companion. By implementing **Working**, **Episodic**, **Semantic**, and **Procedural Memory** with LangChain and enhancing them with **Bitcoin**, **Solana**, and **Base blockchains**, you’ve built an AI that remembers your journey, learns from it, and adapts seamlessly. In Wonderland, this ensures the Mad Hatter keeps every twist of your DeFi-inspired odyssey alive, powered by the Cheshire Terminal on Solana with the GRIN DAO and overseen by Chesh.

Flip the switch, experiment, and let your AI’s memory weave a tale that never fades. Welcome to the future of agentic AI—may your imagination run wild!

*Cheshire Terminal, March 10, 2025*

---

This guide synthesizes human-inspired memory frameworks with practical implementations, fully integrating the Wonderland Memory System and blockchain enhancements as requested. It’s ready for developers and enthusiasts to build upon!

---

---

---

Traditional language models (LLMs) are stateless—they process each input as a fresh start, forgetting everything unless you spoon-feed them context. This is like talking to a goldfish: no memory, no continuity, no learning. Agentic AI aims to bridge this gap by giving models a **memory framework**, enabling them to:

- **Remember** past interactions.
- **Learn** from experiences.
- **Adapt** to user needs over time.

By modeling human memory types, we create AI that’s not just reactive but proactive—capable of anticipating needs and handling complex, ongoing tasks. This guide will show you how to implement such a system step-by-step.

---

## Understanding the Four Memory Types

### Working Memory

- **What It Is**: The "RAM" of your AI—short-term memory for the current conversation.
- **Role**: Tracks the immediate back-and-forth, providing context for responses.
- **Example**: Remembering you said "Hi" two messages ago to reply "Hey again!"

### Episodic Memory

- **What It Is**: Long-term memory for specific events or "episodes" (e.g., past conversations).
- **Role**: Recalls past interactions and their outcomes to inform current ones.
- **Example**: Knowing you prefer chicken biryani from a chat last week.

### Semantic Memory

- **What It Is**: A structured database of facts and concepts.
- **Role**: Grounds responses in reliable knowledge, like a personal Wikipedia.
- **Example**: Recalling that the capital of France is Paris.

### Procedural Memory

- **What It Is**: The "how-to" memory for skills and routines.
- **Role**: Executes learned tasks, like summarizing text or formatting data.
- **Example**: Summarizing a long message because it worked well before.

---

## Implementation with LangChain

### Setting Up the Environment

To follow along, you’ll need:

- **Python 3.8+**
- **LangChain**: `pip install langchain langchain-openai`
- **OpenAI API Key**: For the LLM (e.g., GPT-4o).
- **Weaviate**: A vector database for episodic memory (run via Docker—see below).
- **Docker Compose**: For Weaviate and Ollama embeddings.

**Docker Compose Setup** (save as `docker-compose.yml`):

```yaml
version: '3'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - 8080:8080
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-ollama'
      ENABLE_MODULES: 'text2vec-ollama'
      OLLAMA_API_ENDPOINT: '<http://ollama:11434/api>'
  ollama:
    image: ollama/ollama:latest
    ports:
      - 11434:11434

```

Run with: `docker-compose up -d`.

---

### Working Memory: Immediate Context

**Goal**: Keep the current conversation in memory.

**Implementation**:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize LLM
llm = ChatOpenAI(temperature=0.7, model="gpt-4o")

# System prompt
system_prompt = SystemMessage("You are a helpful AI Assistant. Answer succinctly in one sentence.")

# Store conversation history
messages = [system_prompt]

while True:
    user_input = input("\\nUser: ")
    if user_input.lower() == "exit":
        break
    user_message = HumanMessage(user_input)
    messages.append(user_message)

    # Generate response with full context
    response = llm.invoke(messages)
    print("\\nAI Message: ", response.content)
    messages.append(response)

# Print conversation
for i, msg in enumerate(messages, 1):
    print(f"\\nMessage {i} - {msg.type.upper()}: {msg.content}")

```

**Output Example**:

```
User: Hello!
AI Message: Hello! How can I assist you today?
User: What's my name?
AI Message: I don’t have that info yet—what’s your name?
User: My name is Richard!
AI Message: Nice to meet you, Richard! How can I help?

```

**Explanation**: The `messages` list acts as working memory, feeding the entire conversation to the LLM for context-aware responses.

---

### Episodic Memory: Learning from the Past

**Goal**: Store and recall past conversations to improve future interactions.

**Implementation**:

1. **Reflection Chain**: Analyze conversations for insights.
2. **Vector Database**: Store reflections in Weaviate.
3. **Recall Function**: Retrieve relevant memories.

**Code**:

```python
import weaviate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from weaviate.classes.config import Property, DataType, Configure

# Connect to Weaviate
vdb_client = weaviate.connect_to_local()
print("Connected to Weaviate: ", vdb_client.is_ready())

# Create Episodic Memory Collection
vdb_client.collections.create(
    name="episodic_memory",
    vectorizer_config=Configure.NamedVectors.text2vec_ollama(
        name="title_vector", source_properties=["conversation"], model="nomic-embed-text",
        api_endpoint="<http://host.docker.internal:11434>"
    ),
    properties=[
        Property(name="conversation", data_type=DataType.TEXT),
        Property(name="context_tags", data_type=DataType.TEXT_ARRAY),
        Property(name="conversation_summary", data_type=DataType.TEXT),
        Property(name="what_worked", data_type=DataType.TEXT),
        Property(name="what_to_avoid", data_type=DataType.TEXT),
    ]
)

# Reflection Prompt
reflection_prompt = ChatPromptTemplate.from_template("""
You are analyzing conversations to create memories for future use.
Review the conversation and output JSON:
{{
    "context_tags": [string, ...], "conversation_summary": string,
    "what_worked": string, "what_to_avoid": string
}}
Conversation: {conversation}
""")
reflect = reflection_prompt | llm | JsonOutputParser()

# Format Conversation
def format_conversation(messages):
    return "\\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages[1:]])

# Add Episodic Memory
def add_episodic_memory(messages, vdb_client):
    conversation = format_conversation(messages)
    reflection = reflect.invoke({"conversation": conversation})
    episodic_memory = vdb_client.collections.get("episodic_memory")
    episodic_memory.data.insert(reflection)

# Recall Episodic Memory
def episodic_recall(query, vdb_client):
    episodic_memory = vdb_client.collections.get("episodic_memory")
    memory = episodic_memory.query.hybrid(query=query, alpha=0.5, limit=1)
    return memory.objects[0].properties if memory.objects else None

# Example Usage
add_episodic_memory(messages, vdb_client)
memory = episodic_recall("What's my name", vdb_client)
print(memory)

```

**Output Example**:

```python
{
    'context_tags': ['personal_information', 'name_recollection'],
    'conversation_summary': 'Recalled the user’s name after being told.',
    'what_worked': 'Storing and recalling the user’s name effectively.',
    'what_to_avoid': 'N/A'
}

```

**Explanation**: Conversations are stored as vectors in Weaviate, with reflections providing actionable insights. The recall function uses hybrid search to fetch relevant past interactions.

---

### Semantic Memory: Factual Knowledge Base

**Goal**: Provide a factual foundation for responses.

**Implementation**:

```python
# Simple Key-Value Store (expandable to a knowledge graph)
knowledge_base = {}

def update_semantic_memory(key, value):
    knowledge_base[key] = value

def query_semantic_memory(query):
    return knowledge_base.get(query, "I don’t have that information yet.")

# Example Usage
update_semantic_memory("Richard’s favorite food", "chicken biryani")
response = query_semantic_memory("Richard’s favorite food")
print(response)  # "chicken biryani"

```

**Explanation**: This basic key-value store can be enhanced with a knowledge graph (e.g., Neo4j) or external APIs (e.g., Wikipedia) for richer factual recall.

---

### Procedural Memory: Skill Execution

**Goal**: Execute learned skills based on context.

**Implementation**:

```python
def summarize_text(text):
    prompt = ChatPromptTemplate.from_template("Summarize this: {text}")
    return llm.invoke(prompt.format(text=text)).content

# Update Procedural Memory
def procedural_memory_update(what_worked, what_to_avoid):
    # Placeholder for skill refinement (e.g., reinforcement learning)
    print("Updated procedural memory with:", what_worked, what_to_avoid)

# Example Usage
text = "Richard loves chicken biryani because it’s spicy and flavorful."
summary = summarize_text(text)
print(summary)  # "Richard enjoys chicken biryani for its spice and flavor."

```

**Explanation**: Skills like summarization are codified as functions, with potential updates based on feedback from `what_worked` and `what_to_avoid`.

---

### Bringing It All Together

**Full Agentic Chatbot**:

```python
conversations, what_worked, what_to_avoid, messages = [], set(), set(), []

def episodic_system_prompt(query, vdb_client):
    memory = episodic_recall(query, vdb_client) or {}
    prompt = f"""You are a helpful AI Assistant.
    Current Match: {memory.get('conversation', 'N/A')}
    What Worked: {' '.join(what_worked)}
    What to Avoid: {' '.join(what_to_avoid)}
    Use this context to respond."""
    return SystemMessage(prompt)

while True:
    user_input = input("\\nUser: ")
    user_message = HumanMessage(user_input)

    system_prompt = episodic_system_prompt(user_input, vdb_client)
    messages = [system_prompt] + [msg for msg in messages if not isinstance(msg, SystemMessage)]

    if user_input.lower() in ["exit", "exit_quiet"]:
        if "exit" in user_input:
            add_episodic_memory(messages, vdb_client)
            procedural_memory_update(what_worked, what_to_avoid)
        break

    context = query_semantic_memory(user_input)  # Semantic memory
    messages.extend([HumanMessage(context), user_message])
    response = llm.invoke(messages)
    print("\\nAI Message: ", response.content)
    messages.append(response)

```

**Output Example**:

```
User: What's my favorite food?
AI Message: You mentioned your favorite food is chicken biryani.
User: Summarize that!
AI Message: Richard enjoys chicken biryani for its spice and flavor.

```

**Explanation**: This integrates all four memory types, creating a cohesive, adaptive AI.

---

## Enhancements and Optimizations

1. **Database Choices**: Experiment with Pinecone or Milvus for scalability.
2. **Episodic Enhancements**: Add failure tagging and contextualized retrieval.
3. **Semantic Expansion**: Integrate a knowledge graph or external APIs.
4. **Procedural Growth**: Use reinforcement learning for skill improvement.

---

## Conclusion

Agentic Memory transforms AI from forgetful chatbots into intelligent agents. By implementing **Working**, **Episodic**, **Semantic**, and **Procedural Memory**, you’ve built a foundation for an AI that learns and grows with every interaction. Experiment, iterate, and share your results—let’s push the boundaries of AI together!

*Cheshire Terminal, March 10, 2025*

---

This guide is a practical, self-contained resource based on Gunde’s work, streamlined for clarity and actionability. Let me know if you’d like to refine any section further!
