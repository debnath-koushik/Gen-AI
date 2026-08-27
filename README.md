# Again Starting From Zero

A beginner-friendly Generative AI book, written from this repo.

This is **not** a production product. It is a learning path: theory from the 33-page Sheryians AI School notes (LLMs, providers, LangChain, models, prompts, structured output, chains, memory, indexes/RAG, agents), plus the real experiments, errors, and fixes in this project.

If you are new, read top to bottom. If you already coded here, jump to the chapter for the file you forgot.

**Practical path in this repo:**

Mistral → Hugging Face → PromptTemplate / ChatPromptTemplate → PydanticOutputParser → structured movie extraction → Streamlit

---

## How to read this

| I want… | Go here |
|---|---|
| The idea of GenAI in plain language | [Part A — Theory](#part-a--theory-from-the-notes) |
| Folder map and what each file is for | [The map of the repo](#the-map-of-the-repo) |
| Setup and run commands | [Setup](#setup) then [How to run each file](#how-to-run-each-file) |
| Hands-on chapters (the actual code) | [Part B — Practice](#part-b--practice-the-code-in-this-repo) |
| Bugs I hit and how they were fixed | [Errors, bugs, and fixes](#errors-bugs-and-fixes) |
| Topics in the notes that I have not coded yet | [Part C — Not built yet](#part-c--topics-in-the-notes-not-built-in-code-yet) |

---

## Why this repo exists

The folder name says it: **Again Starting From Zero**.

The notes teach a full GenAI stack. This repo rebuilds it from the smallest possible experiment, instead of copying a finished RAG app:

1. Can I call a model at all?
2. Can I switch providers without rewriting everything?
3. Can I run a model on my own machine?
4. Can I turn text into numbers (embeddings)?
5. Can I hold a conversation, not just one question?
6. Can I give the bot a personality?
7. Can I put a UI on it?
8. Can I control the *shape* of the answer with a prompt template?
9. Can I force that answer into a real Python object (Pydantic)?

Older ideas are often left commented in the same file. That is on purpose. The comments *are* the lesson.

---

## The map of the repo

```
Again_Starting_From_Zero/
├── test.py                         # Does my Hugging Face token work?
├── requirements.txt                # Libraries (some are for later: RAG, FastAPI)
├── .env                            # API keys — never commit this
├── .gitignore
│
├── chat_models/                    # MODELS
│   ├── chat_model.py               # Unique initializer: ChatMistralAI
│   ├── init_chat_model.py          # Universal initializer: init_chat_model
│   └── localmodel_by_hf.py         # TinyLlama running locally
│
├── embedding_models/               # EMBEDDINGS (needed later for RAG)
│   └── embeddings.py
│
└── project/                        # APPLICATIONS
    ├── chatbot.py                  # Terminal chatbot (the memory diary)
    ├── ui_chatbot.py               # Streamlit chat, funny AI
    ├── ui_chatbot2.py              # Streamlit chat + personality picker
    ├── PromptTemplate.py           # Terminal: movie extract as formatted text
    ├── ui_prompt_template.py       # Same extract, Streamlit UI
    ├── structured_output.py        # Terminal: PydanticOutputParser → Movie object
    ├── ui_structured_output.py     # Streamlit: parser in prompt, still shows raw text
    └── ui_structured_output2.py    # Streamlit: parse + show JSON (the complete version)
```

**Theory → files**

| Notes topic | Coded here? | File |
|---|---|---|
| What is an LLM | Yes (explained + used) | all chat files |
| Providers (OpenAI / Gemini / Claude / …) | Yes, as Mistral + Hugging Face | `chat_models/` |
| Why LangChain | Yes | whole repo |
| Models (chat + embeddings) | Yes | `chat_models/`, `embedding_models/` |
| Prompts (simple, system+user, templates) | Yes | `chatbot.py`, `PromptTemplate.py` |
| Structured output | Yes | `structured_output.py`, UI files |
| Memory | Partial (manual message list) | `chatbot.py`, UI chatbots |
| Chains | Not yet | — |
| Indexes / RAG | Not yet (libs installed) | `embeddings.py` is only the vector step |
| Agents | Not yet | — |

---

## Setup

### What you need

- Python 3.12 (this project's venv used 3.12)
- A `.env` file in the repo root:

```
MISTRAL_API_KEY=your_mistral_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
HF_TOKEN=your_hf_token
```

- Mistral key → `ChatMistralAI` and `MistralAIEmbeddings`
- Hugging Face token → Hub models and local downloads

`.env` is in `.gitignore`. Never put keys in git or in this README.

### Install

```bash
python -m venv .venv
source .venv/Scripts/activate    # Git Bash on Windows
pip install -r requirements.txt
```

`requirements.txt` is wider than the code that exists today. `faiss-cpu`, `chromadb`, `pypdf`, `unstructured`, and FastAPI are **future** tools (RAG and APIs). They are listed because the notes go there next.

### Local cache path (Windows, this machine)

In `localmodel_by_hf.py` and `embeddings.py`:

```python
os.environ['HF_HOME'] = 'D:/Koushik/huggingface_cache'
```

Hugging Face models are large. This keeps them on `D:` instead of filling `C:`. On another computer, change this path or delete the line.

---

# Part A — Theory (from the notes)

These ideas come from the 33-page Generative AI notes (Sheryians AI School). They are rewritten here in beginner language and tied to this repo.

## A1. Before you start

The notes assume you already know a little:

- Python (functions, classes, lists, dicts)
- Machine learning at a high level (train on data, predict)
- Deep learning / transformers at a high level (not “build GPT from scratch”)
- NLP basics: **tokenization** (text → tokens) and **embeddings** (text → numbers)

You do **not** need to train an LLM. In Generative AI we almost never train GPT/Mistral from zero. We **use** pre-built models.

## A2. Four phases of the field (so you know where you stand)

The notes split the industry into layers:

| Phase | Who lives here | What they do |
|---|---|---|
| 1. Research | Labs (OpenAI, Google, Mistral, Meta, …) | Invent and train huge models |
| 2. Bedrock models | The actual LLMs | GPT, Gemini, Claude, LLaMA, Mistral, Grok, … |
| 3. Providers | Cloud APIs and hubs | Give you a key and an HTTP endpoint |
| 4. Application | **You** | Use those models to build products |

This repo is **Phase 4**. We do not train Mistral. We call it, prompt it, remember conversation, and structure its output.

The notes' long-term plan of action is:

1. LLM foundation ← **you are here**
2. RAG systems ← next
3. Agentic AI
4. Project and deployment

## A3. What is an LLM?

**LLM = Large Language Model.** Break the three words:

| Word | Meaning |
|---|---|
| **Large** | Trained on a huge pile of text (books, websites, code, chats, …) |
| **Language** | It reads and writes human language — and code |
| **Model** | A neural network that learned patterns from that data |

When you type `"What is Python?"` the model does **not**:

- search Google
- think like a person
- “understand” the way you do

It **predicts the next likely token**, then the next, then the next. That is why prompts matter: you are steering a next-token machine, not briefing a colleague.

Popular families: GPT, Gemini, LLaMA, Claude, Grok, Mistral, …

## A4. The provider problem (why LangChain exists)

Each company has its own SDK and response shape.

The notes show three raw styles:

- **OpenAI:** `client.chat.completions.create(...)` then `response.choices[0].message.content`
- **Gemini:** `genai.GenerativeModel(...).generate_content(...)` then `response.text`
- **Claude:** `client.messages.create(...)` then `response.content[0].text`

If every file in your app talks to one SDK, switching vendor means rewriting everything.

**LangChain is the adapter.** You build against `model.invoke(...)`. The provider sits behind a class (`ChatMistralAI`, `ChatHuggingFace`, or `init_chat_model`).

This project uses **Mistral** as the main API and **Hugging Face** for Hub / local models. Same idea as the notes' OpenAI/Gemini/Claude examples.

## A5. LangChain's six components

From the notes. Remember this diagram. The whole course is these six boxes.

```
┌─────────┐   ┌─────────┐   ┌─────────┐
│  Model  │   │ Prompt  │   │  Chain  │
└─────────┘   └─────────┘   └─────────┘
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Memory  │   │ Indexes │   │  Agent  │
└─────────┘   └─────────┘   └─────────┘
```

### Model

The intelligence layer. Three types you will actually use:

| Type | Job | In this repo |
|---|---|---|
| **Chat / language models** | Text in → text out (answers, code, summaries) | `ChatMistralAI`, TinyLlama |
| **Embedding models** | Text in → vector (list of floats) | `MistralAIEmbeddings`, MiniLM |
| **Multimodal models** | Text + images / audio / files | not used yet |

### Prompt

The instruction. It tells the model **what to do, how to respond, and in what format**.

Without a prompt, the model has no job. Better prompt → better output. LLMs are not mind readers.

Types you will use:

1. **Simple prompt** — `"Explain machine learning."` (see `chat_model.py`)
2. **System + user prompt** — system sets behaviour, user asks the question (see `chatbot.py`)
3. **Prompt templates** — reusable text with `{variables}` (see `PromptTemplate.py`)
4. **Structured output** — force JSON / a schema so a program can read it (see `structured_output.py`)

### Chain

A **pipeline** of steps, not one prompt → one reply.

Example from the notes: “Summarize this article and translate it into Hindi” is really:

1. summarize
2. translate the summary
3. return the result

This repo has not built LangChain chains yet. You still call `model.invoke()` once.

### Memory

LLMs are stateless. Each API call is independent unless **you** send previous messages again.

- Without memory: “My name is Akarsh.” later “What is my name?” → it does not know.
- With memory: you resend the history.

This repo does memory by hand with a `messages` list (and Streamlit `session_state`). It does not yet use LangChain memory classes, trimming, or summarization.

### Indexes (RAG)

The model only knows training data. It does not know your PDFs, company wiki, or private DB.

**Indexes** connect external data. The usual pattern is **RAG** (Retrieval Augmented Generation):

1. Load documents
2. Split into chunks
3. Embed chunks → vectors
4. Store in a vector DB (FAISS, Chroma, …)
5. Embed the user question
6. Retrieve nearest chunks
7. Put those chunks in the prompt
8. Let the chat model answer from that context

This repo only does step 3 in `embeddings.py`. RAG is the next big chapter.

### Agent

A **chain** follows a fixed recipe. An **agent** decides the next step.

A plain LLM can only generate text. It cannot search the web, run a calculator, hit an API, or query a database unless you give it **tools** and let it choose them. That is agents. Not built here yet.

---

# Part B — Practice (the code in this repo)

Read these in order the first time. Each chapter is one idea.

## Chapter 1 — First contact: is the setup alive?

**Files:** `test.py`, then `chat_models/chat_model.py`

`test.py` is not a chatbot. It only asks Hugging Face “who am I with this token?”

```python
from huggingface_hub import whoami
print(whoami(token=os.getenv("HUGGINGFACEHUB_API_TOKEN")))
```

If this fails, nothing else will work. Fix `.env` first.

Then the smallest real LLM loop in `chat_model.py`:

```python
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(
    model="mistral-small-latest",  # also tried: mistral-small-2506
    temperature=0.7,
    max_tokens=1024,
)

response = model.invoke("What is the national animal of India?")
print(response.content)
```

**The loop you will see everywhere:**

load secrets → create model → `invoke()` → read `response.content`

| Knob | Meaning |
|---|---|
| `temperature=0.7` | Higher = more variety. Lower = more repeatable. 0.7 is conversational. |
| `max_tokens=1024` | Cap on reply length (cost + runaway text). |
| `invoke()` | One request, one response. No streaming, no tools. |
| `response.content` | The text. The rest of `response` is metadata. |

Hugging Face chat (`ChatHuggingFace` + `HuggingFaceEndpoint`) is commented in this file. Both providers were tried. **Mistral became the default** for later apps because the chat path was simpler.

Comment left in the file:

> Model class LangChain ka unique initializer hai. Jab koi unique model ke upar hum ek project bnana chahete hein tab hum iska use karte hein.

Meaning: `ChatMistralAI` is locked to Mistral. Use it when the project will stay on that provider.

---

## Chapter 2 — Two ways to create a model

**Files:** `chat_model.py` vs `init_chat_model.py`

This confusion is why both files exist.

### Unique initializer — stay on one vendor

```python
from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model="mistral-small-latest", ...)
```

You import a **provider package**. Fine when Mistral is the decision.

### Universal initializer — vendor might change

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "Qwen/Qwen3-4B",
    model_provider="huggingface",
)
```

Comment in the file:

> init_chat_model LangChain ka universal initializer hai. Jab model kabhi bhi change ho sakta ho tab hum iska use karte hein.

Same `.invoke()` either way. Construction is what changes.

Later files mostly use `ChatMistralAI` because the learning goal moved from “swap providers” to “chat, prompts, structure.”

---

## Chapter 3 — Run a model on this machine

**File:** `chat_models/localmodel_by_hf.py`

No Mistral API. TinyLlama downloads into `D:/Koushik/huggingface_cache` and runs with `transformers` + PyTorch.

```python
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)
chat_model = ChatHuggingFace(llm=llm)
response = chat_model.invoke("What is 2 + 2?")
```

TinyLlama (1.1B) is small enough for a laptop. A 30B cloud model is not.

| | API (Mistral) | Local (TinyLlama) |
|---|---|---|
| Cost | Pay per token | Free after download |
| Quality | Stronger | Weaker, good for plumbing |
| Setup | API key | Disk, RAM, CPU/GPU, `torch` |
| Privacy | Text leaves the machine | Text stays here |
| First run | Instant | Slow (download) |

- `do_sample=False` — greedy decoding, more stable answers
- `repetition_penalty=1.03` — slightly less looping
- `ChatHuggingFace` — wrap the raw pipeline so you still call `.invoke()` like a chat model

---

## Chapter 4 — Embeddings: text becomes numbers

**File:** `embedding_models/embeddings.py`

Chat models answer in language. Embedding models answer in **vectors**.

```python
embedding = MistralAIEmbeddings(model="mistral-embed")

texts = [
    "Hello this is my Machine",
    "I am using vscode for coding",
]
vector = embedding.embed_documents(texts)
print(vector)
```

Hugging Face version is commented: `sentence-transformers/all-MiniLM-L6-v2` (small, can run locally).

**Mental model**

- Chat model: text → text
- Embedding model: text → list of floats
- Similar meaning → vectors that sit close together

This is the missing piece for RAG (indexes in the notes). This file only *prints* vectors so you can see that text became numbers. No vector database yet, even though FAISS and Chroma are in `requirements.txt`.

---

## Chapter 5 — Chatbot as a diary (memory)

**File:** `project/chatbot.py`

This is the most important file. Old versions are commented in, not deleted.

The live model is pinned: `mistral-small-2506` (not `latest`). Pinning means behaviour does not silently change when the vendor updates `latest`.

### Version 1 — one question

```python
prompt = input("User: ")
response = model.invoke(prompt)
print("AI: ", response.content)
```

No memory. Follow-ups like “what about its habitat?” fail because the model never saw “it”.

### Version 2 — loop, still stateless

```python
while True:
    prompt = input("User: ")
    if prompt == "0":
        break
    response = model.invoke(prompt)
    print("AI: ", response.content)
```

Many questions, each independent. `0` still means quit.

### Version 3 — naive memory (raw strings)

```python
messages = []
messages.append(prompt)
response = model.invoke(messages)
messages.append(response.content)
```

Follow-ups start working. Then the comments list why this is still a bad design — that list *is* the memory chapter of the notes, written from practice:

- no role separation (system / user / assistant)
- raw strings → weak structure
- history grows forever
- hits the token limit
- cost goes up
- slower replies
- no trimming
- no summarization
- not production-scale
- no control of the context window

### Version 4 — real message roles

From the notes: modern chat = **system + user**. LangChain names:

| Class | Who | Why |
|---|---|---|
| `SystemMessage` | Hidden instructions | Personality and rules |
| `HumanMessage` | You | What was typed |
| `AIMessage` | The model | What it already said |

```python
messages = [SystemMessage(content="You are a funny AI agent")]
messages.append(HumanMessage(content=prompt))
response = model.invoke(messages)
messages.append(AIMessage(content=response.content))
```

Shape of a real chat:

```
[system] You are a funny AI agent
[human]  Hello
[ai]     Hey! What's up?
[human]  Tell me a joke about Python
```

### Version 5 — personality is a system prompt (live code)

Menu:

1. Professional — polished and precise
2. Friendly — warm and chatty
3. Cynical — critical and sarcastic

Same model. Different `SystemMessage`. Personality is **not** a different LLM.

At the end, `print(messages)` dumps the transcript as objects — useful when debugging “why did it forget?” or “why is it sarcastic?”

There was a bug here (`elif choice == "1"` twice). It is **fixed** now (`elif choice == "2"`). See [Errors, bugs, and fixes](#errors-bugs-and-fixes).

---

## Chapter 6 — Streamlit: a face on the same chat

**File:** `project/ui_chatbot.py`

```bash
streamlit run project/ui_chatbot.py
```

The model call did not change. The UI did.

Streamlit **re-runs the whole script** on every click/key. A normal `messages = []` would wipe history every turn. That is why chat lives in `st.session_state.messages`.

Also:

- `st.chat_input` / `st.chat_message` — bubbles instead of terminal I/O
- `SystemMessage` is **not** drawn on screen (the user should not see “You are a funny AI agent”)
- personality is hardcoded: funny AI

UI is display + storage. GenAI is still `model.invoke(messages)`.

---

## Chapter 7 — Personality in the UI

**File:** `project/ui_chatbot2.py`

`ui_chatbot.py` + the personality idea from `chatbot.py`, done properly.

- Sidebar radio: Professional / Friendly / Cynical
- Each maps to a `system_prompt`
- Changing personality **resets** history so the old persona does not leak

```python
if (
    "messages" not in st.session_state
    or st.session_state.get("current_personality") != personality
):
    st.session_state.current_personality = personality
    st.session_state.messages = [SystemMessage(content=system_prompt)]
```

That `current_personality != personality` check is the important line.

Lessons:

- system prompt = behaviour
- `session_state` = memory across Streamlit reruns
- reset history when the system prompt changes

Still not production: no login, no database, no token trim, no streaming.

---

## Chapter 8 — Prompt templates (movie extract as text)

**Files:** `project/PromptTemplate.py` (terminal), `project/ui_prompt_template.py` (Streamlit)

Until now the user typed free English. Now the job is: **paragraph in → filled form out**.

Comment at the top of the file (learn this distinction):

| Class | Use |
|---|---|
| `PromptTemplate` | Simple text with `{variables}` — `"Say {foo}"` |
| `ChatPromptTemplate` | Chat roles: system / human / AI — matches chat models |

This project uses `ChatPromptTemplate.from_messages`:

1. **System** — rules + exact output headings (Movie Name, Year, Director, …)
2. **Human** — `"Here is the paragraph to analyze: {paragraph}"`

```python
paragraph = input("Give your paragraph: ")
final_prompt = prompt.invoke({"paragraph": paragraph})
response = model.invoke(final_prompt)
print(response.content)
```

`prompt.invoke(...)` fills `{paragraph}`. Then the model runs. The new skill is **build the prompt separately from calling the model**.

The system rules are strict on purpose:

- use ONLY the paragraph
- do not invent
- missing field → `"Not mentioned"`
- keep a fixed layout

Without that, the model “helps” with training-data facts. That is **hallucination**. For extraction, hallucination is a failed program.

`ui_prompt_template.py` is the same prompt with a text area and an Extract button. Output is still **plain text**, not a Python object.

---

## Chapter 9 — Structured output (Pydantic)

**Files:**

| File | What it does |
|---|---|
| `project/structured_output.py` | Terminal: schema + parser + `parse()` |
| `project/ui_structured_output.py` | Streamlit: schema in the prompt, still prints raw LLM text |
| `project/ui_structured_output2.py` | Streamlit: `parse()` + `st.json(...)` — the complete version |

Chapter 8 asked the model to *look like* a form. Chapter 9 asks for a **typed object** a program can use (`movie.title`, `movie.cast`, …).

### The schema

```python
from pydantic import BaseModel
from typing import List, Optional

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str
```

`Optional` means “this field may be missing.” Lists mean “zero or more.”

### The parser

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Movie)
```

`parser.get_format_instructions()` generates the JSON schema text that goes **into the prompt**, so the model knows the exact shape.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract movie information from the paragraph\n{format_instructions}"),
    ("human", "Here is the paragraph to analyze:\n{paragraph}"),
])
```

### The two-step call

```python
final_prompt = prompt.invoke({
    "paragraph": paragraph,
    "format_instructions": parser.get_format_instructions(),
})
response = model.invoke(final_prompt)
movie_data = parser.parse(response.content)  # str → Movie
print(movie_data)
```

Flow:

```
paragraph
    → ChatPromptTemplate (+ format instructions)
    → LLM (JSON-looking text)
    → PydanticOutputParser.parse
    → Movie object
```

`ui_structured_output.py` stops before `parse()` and shows `response.content`. That was a midpoint: you can see the raw JSON the model returned.

`ui_structured_output2.py` is the goal:

```python
movie_data = parser.parse(response.content)
st.session_state.movie_data = movie_data
st.json(movie_data.model_dump())
```

`model_dump()` turns the Pydantic object into a dict for the JSON view.

There is a typo in the **terminal** file (`format_instrucions`). The UI files spell it correctly. Details in [Errors, bugs, and fixes](#errors-bugs-and-fixes).

---

## The concepts that keep repeating

If you remember only this table, you remember the course so far.

| Concept | One sentence | Where |
|---|---|---|
| `.env` + `load_dotenv()` | Secrets stay out of code | every API file |
| `invoke()` | “Run the model now” | everywhere |
| Unique vs universal init | Locked vendor vs swap-friendly factory | `chat_model.py` vs `init_chat_model.py` |
| API vs local | Pay-and-go vs download-and-run | Mistral vs TinyLlama |
| `temperature` | Randomness of the answer | chat files |
| Embeddings | Meaning as coordinates | `embeddings.py` |
| Stateless vs memory | One shot vs sending history every time | `chatbot.py` |
| Message roles | System / Human / AI is the real chat format | chatbot + UI chats |
| System prompt | Personality and rules live here | chatbot, UI 2, extractors |
| Streamlit rerun | Script restarts; only `session_state` survives | all `ui_*.py` |
| Prompt templates | Fill `{variables}`, then invoke | `PromptTemplate.py` |
| Structured output | Schema + parser → Python object | `structured_output.py`, `ui_structured_output2.py` |

---

# Errors, bugs, and fixes

These are real mistakes from this project. Read them so you do not “debug mystery GenAI” when it is just Python.

### 1. Friendly personality never ran (`chatbot.py`) — FIXED

**What happened:** both branches checked `"1"`.

```python
if choice == "1":
    mode = "You are a Professional AI. ..."
elif choice == "1":   # bug: should be "2"
    mode = "You are a Friendly AI. ..."
```

Typing `2` fell through to Cynical.

**Fix:** `elif choice == "2":` — this is the current code. `ui_chatbot2.py` never had this bug (it uses radio labels, not `"1"`/`"2"`).

### 2. Typo: `format_instrucions` (`structured_output.py`) — still there in the terminal file

**What happened:** the template variable and the dict key were misspelled (`instrucions`).

```python
# system prompt uses {format_instrucions}
final_prompt = prompt.invoke({
    "paragraph": paragraph,
    "format_instrucions": parser.get_format_instructions(),  # same typo, so it still fills
})
```

Because **both** the template and the dict use the same wrong spelling, the terminal script can still run. It is a trap: copy the prompt to a UI that uses `{format_instructions}` and forget to rename the key → the model never sees the schema.

**Fix in the UI files:** `{format_instructions}` + key `"format_instructions"`. Prefer that spelling everywhere. The terminal file is left as a reminder of the typo.

### 3. Parser created but not used (`ui_structured_output.py`)

The file builds `PydanticOutputParser` and injects format instructions, then does:

```python
st.write(response.content)  # raw string, not Movie
```

That is a useful midpoint (inspect the JSON), not the end.

**Complete version:** `ui_structured_output2.py` calls `parser.parse(response.content)` then `st.json(movie_data.model_dump())`.

If `parse()` crashes, the model did not return valid JSON/schema. Typical causes: temperature too high, weak model, paragraph too messy, or format instructions never reached the prompt (see typo above).

### 4. Hugging Face / Mistral “it just fails”

Checklist before blaming LangChain:

1. `.env` exists in the **repo root** and `load_dotenv()` runs
2. Key names match exactly: `MISTRAL_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`
3. `python test.py` works for Hugging Face
4. First local TinyLlama run needs disk + time (download)
5. `HF_HOME` on another PC may point at a path that does not exist

### 5. Streamlit “forgot” the chat

If history dies after one message, you stored `messages` in a normal variable. Use `st.session_state`. If personality changes but old jokes remain, you did not reset messages when the system prompt changed (`ui_chatbot2.py`).

### 6. Hallucinated movie facts

If the extractor invents a director or year, the prompt is too loose. Chapter 8’s “ONLY this paragraph / Not mentioned” rules exist for that. Structured output still needs the model to cooperate; Pydantic only **validates** the shape, it cannot know if a year is true.

---

# Part C — Topics in the notes, not built in code yet

The PDF goes further than this repo. Here is what those chapters mean, so a beginner is not lost when they appear in videos/notes.

## Chains (notes, no file yet)

Today: one prompt → one `invoke()`.

A chain is several steps glued together, for example:

```
prompt | model | parser
```

or: summarize → translate → return.

When this is added, the movie extractor can become a real pipeline instead of three manual Python lines.

## Memory (partial)

Done: send a growing `messages` list.

Not done (already listed as problems in `chatbot.py`):

- trim old turns
- summarize old chat into one `SystemMessage`
- persist to a database
- LangChain memory helpers

Without trimming, long chats hit the context window, get slower, and cost more.

## Indexes / RAG (notes, no pipeline yet)

Done: embed a couple of sentences.

Not done: load PDF → split → vector store → retrieve → answer.

That is how the model would answer from *your* documents instead of only training data.

Libraries already in `requirements.txt`: `faiss-cpu`, `chromadb`, `pypdf`, `unstructured`.

## Agents (notes, no file yet)

A chain is a recipe. An agent **chooses tools** (search, calculator, API, SQL).

Do not start agents until memory and RAG feel boring. The notes' order is: foundation → RAG → agents → deploy.

## Other gaps

- streaming tokens in the UI
- FastAPI (installed, unused)
- one shared model config (today every file recreates `ChatMistralAI(...)`)
- tests and error handling (missing key, rate limit, parse failure)

---

## How to run each file

From the repo root, venv active, `.env` filled.

```bash
# Auth check
python test.py

# Models
python chat_models/chat_model.py
python chat_models/init_chat_model.py
python chat_models/localmodel_by_hf.py          # first run downloads TinyLlama

# Embeddings
python embedding_models/embeddings.py

# Chat (terminal: pick 1/2/3, then chat; press 0 to quit)
python project/chatbot.py

# Chat UI
streamlit run project/ui_chatbot.py
streamlit run project/ui_chatbot2.py

# Prompt template extract (text form)
python project/PromptTemplate.py
streamlit run project/ui_prompt_template.py

# Structured output (Pydantic)
python project/structured_output.py
streamlit run project/ui_structured_output.py     # raw model text
streamlit run project/ui_structured_output2.py    # parsed JSON
```

Try a movie paragraph like:

> Inception (2010), directed by Christopher Nolan, is a science-fiction thriller starring Leonardo DiCaprio. It follows a thief who enters people's dreams.

You should see fields filled from the text, and `"Not mentioned"` (or `null` in JSON) for things the paragraph never said.

---

## Timeline

| When | What |
|---|---|
| 3 Aug 2026 | First commit: env, three model experiments, embeddings, one-shot chatbot |
| 26 Aug 2026 | Chat loop, history, roles, personality, Streamlit chat UIs, PromptTemplate started |
| After that | Movie extract prompt, Streamlit extract UI, Pydantic structured output UIs; personality bug fixed |

The repo grew in the same order as the notes: **models → prompts → memory → structured output**. Chains, RAG, and agents are still ahead.

---

## How to use this file later

1. Skim [LangChain's six components](#a5-langchains-six-components).
2. Open the file for the chapter you forgot. Comments inside `chatbot.py` are part of the book.
3. Run that file once so the idea is in your hands again.
4. If something breaks, read [Errors, bugs, and fixes](#errors-bugs-and-fixes) before changing the prompt randomly.

Starting from zero was the point. This README is so you (or any beginner) do not have to start from zero twice.
