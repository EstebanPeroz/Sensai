## Integrating Ollama / MCP-style tools

Tools are wired in directly through the API, via the `tools` field of the
request payload. If the model decides — based on a tool's name,
description, and parameters — that it needs a tool, it stops generating and
returns the request in the message's `tool_calls` field. The caller reads
that, executes the matching function, and sends the return value back as a
new message with role `"tool"`. The model reads that result and continues.
(Reference: <https://docs.ollama.com/capabilities/tool-calling>.)

For models that **don't** support native tool calling, the only option is
to implement the same mechanism via the system prompt: describe a content
format the model should use to signal what it needs, then parse the
model's normal `content` output for that format instead of reading
`tool_calls`. This is explicitly called out as much less reliable — it's a
hand-rolled protocol, so there's real hallucination risk — but it's the
only way to get tool-like behavior out of a model without native support.

**Implementation.** Sensai uses the native mechanism only:

- A tool implements `Tool` (`tools/tool.py`): a `name`, a `description` and a
  JSON schema of its `parameters`, from which the base class builds the
  `tool_rule` sent in the `tools` field, plus a `run` method.
- `ToolManager.call` (`tools/tool_manager.py`) runs the tool named in a
  `tool_calls` entry and always returns a `tool` message. Failures (unknown
  tool, missing or wrong argument, tool error) come back as a content
  starting with `error:`, so the model can read them and react instead of
  the loop crashing.
- Whether a model supports tools is read from the `capabilities` returned by
  `/api/show`. A model without the `tools` capability gets no tools: the
  prompt-based fallback described above is not implemented.

## The reasoning loop (ReAct)

ReAct = reflection + action, combined. After a user prompt, the model
enters:

```
reflect -> act -> observe result -> reflect -> act -> observe result -> ...
```

The hard part, called out explicitly in the notes, is **detecting when
reflection has reached the expected result** — i.e. the loop's termination
condition.

### Implementation (`llm/thinking/react.py`)

Ollama's own thinking is turned off (`"think": false`) so that every model,
with or without that capability, reasons through the same loop. The class
`ReAct` is built by the core and injected into the conversation, which calls
`generate_answer(messages)` and streams what it yields to the UI.

Each iteration sends up to two requests:

1. **Think** — no tools are offered. An instruction asks for two or three
   short sentences of reasoning, ending with `Tool needed: yes` or
   `Tool needed: no` on the first iteration (the instruction then lists the
   tools), and with `Answer found: yes` or `Answer found: no` after a tool
   result. The thought is streamed to the UI as thinking.
2. **Act** — the native tools are offered only if the thought asks for them.
   Otherwise the model is told to write its final answer, which ends the loop.
3. **Observe** — each tool call runs through `ToolManager.call`, its result is
   added as a `tool` message, and the UI shows `-> Using <tool>...`.

Instructions are sent as a last `user` message added to a copy of the
messages, for one request only: they never reach the history. Sending them
as `system` messages at the end of the conversation left artifacts in the
replies of small models (`assistant`, `/no_think`).

Termination:

- the loop ends as soon as the model answers without calling a tool;
- after a tool result, tools are offered again only if the thought ends with
  `Answer found: no`. Stopping is the default because small models otherwise
  keep repeating the same search;
- after `max_iterations` iterations, a last request without tools forces an
  answer.

Only the final answer is saved in the conversation history: thoughts, tool
calls and tool results stay in the message list of the current answer.

### Thinking effort

`/effort off|low|medium|high` sets `max_iterations` to 0, 1, 3 or 5, and the
current level is shown at the bottom right of the input box. `off` skips the
loop entirely: a single request, with neither thinking nor tools. The effort
is a cap, not a target: the model may stop earlier.

### Known limitations

- Only the last decision of a thought counts, read case-insensitively and
  through markdown (`**Tool needed:** No`). Models often copy both options of
  the instruction before deciding; reading any `Tool needed: no` in the
  thought withdrew the tools even when the model went on to say `yes`. A
  thought without any decision offers the tools on the first iteration and
  stops after a tool result.
- Models that always think, such as `qwen3:4b` (the 2507 thinking variant),
  ignore `"think": false`: their reasoning and `</think>` end up in the
  answer, and they are very slow on CPU. Use an instruct variant instead.
- Each iteration costs two requests, so answers are slower.
- Models without the `tools` capability think, but cannot use any tool.
- Nothing prevents the same tool call from being repeated, apart from
  `max_iterations`.

### Alternatives tried (2026-10-12 / 2026-10-13)

Measured with `qwen3:1.7b` and `llama3.2` against fake tools:

- **Text-format ReAct** (`Thought:` / `Action: tool(arg="…")` /
  `Observation:` / `Final Answer:` described in the system prompt, parsed
  from `content`, generation stopped on `Observation:`): one request per
  step and works with every model, but needs a parser, and the models drift
  from the format (`Action: None`, invented tools, no `Final Answer:`).
- **Structured decision** (`format` set to a JSON schema
  `{"thought": string, "need_tool": boolean}` for the Think request): the
  decision was valid on every run and the models chained several searches,
  but they rarely said they were done, so `max_iterations` became the real
  stop.

## Human in the loop

### Confirmation before a tool runs (US-19)

A tool runs only after the user allows it when it declares `requires_confirmation`
(for tools with consequences, such as writing a file), or when it is listed in
`[features].confirm` of the settings (`confirm = ["tools.web_fetch"]`). Nothing is
gated by default.

The check is in `ToolManager.call`, so the model cannot skip it. Before running a
gated tool, it calls the approver given by `Core`, `UIAdapter.ask_confirmation`,
which shows a dialog with the tool name and its arguments:

- **Allow** (`y`): runs this call only.
- **Always this session** (`a`): runs it and stops asking for that tool until Sensai
  restarts. `ToolManager` keeps that list.
- **Deny** (`n`, escape): the tool does not run, and the model receives
  `error: the user declined to run '<name>'; ...` as the tool result, like any other
  tool failure. The loop then answers without it.

Deny is also the answer whenever nobody can be asked: no approver, a dialog that
cannot be shown, or the app closing while the dialog is open.

`/tools list` shows which tools ask for confirmation.

### Asking the user for missing information (US-20)

`ask_user` (`tools/ask_user.py`) is a tool like the others: the model calls it with
one `question` when the request lacks information no other tool can find (which
period, which file). `UIAdapter.ask_user` shows the question in a dialog and returns
the answer, which becomes the tool result, so the loop keeps what it already found.
Escape or an empty answer returns `The user did not answer. State the assumption you
make in your answer.` The question and the answer are also shown in the chat.

It never asks for confirmation, and can be switched off like any tool
(`[features].disabled = ["tools.ask_user"]`, or `/tools disable ask_user`).

### Waiting for the user from the streaming thread

Tools run inside the ReAct generator, so on the streaming thread. The answer cannot
come through the input box: `Conversation.chat` reads any new event in the
`EventQueue` as a new message and stops the current answer. Instead, the Textual
adapter opens a modal screen (`ui/tui/_dialogs.py`) with `call_from_thread` and blocks
the streaming thread on a `threading.Event` until the screen is dismissed. The modal
also covers the input, so no event can interrupt the answer while it waits. The
waiting thread checks every 0.1 s that the app still runs, so it never stays blocked.

### Limitations

- With `/effort off`, or with a model without the `tools` capability, no tool is
  called, so nothing is asked.
- A refused call is not remembered beyond the tool message: the model may call the
  same tool again in the same answer, and the user is asked again.
- The question and the answer of `ask_user` are not saved to the history, like every
  tool exchange: only the final answer is. A later message cannot rely on them.
- Whether `ask_user` is called depends on the model: small models may ask when they
  could answer, or guess instead of asking.

## Detecting hallucination

Proposed approach: **LLM-as-judge**. Send the first model's output to a
second model whose job is to check for hallucinations, giving that second
model tools (web search, RAG) to verify claims against real sources.

## RAG

**Embeddings for RAG:** use `nomic-embed-text` — designed for large context
and a good number of dimensions, and notably fast. `bge-m3` and
`embeddinggemma` are rated as better quality but much slower;
revisiting that tradeoff later if embedding speed turns out to
matter less than expected. Chunks can be embedded in a batch
("simultaneously") rather than one at a time.

**Vector store:** Qdrant, run via Docker.

## Web_search

**Web search:** two categories of API — those needing a key (Ollama's
`web_search` tool, LangSearch) versus keyless tools (`duckduckgo-search`).
The notes call for **a comparison table across content size, relevance, and
speed** before picking one.

**Comparison (2026-10-08):** three pipelines run on 6 real queries (one in
French), each one being a search, then a fetch of the first result. Sizes are
estimated at 4 characters per token, and passages were kept without
embeddings (first chunks of the page).

| Pipeline | Search (median) | Fetch (median) | Raw search output | Raw page (mean) | Answer found after fetch | Search + fetch output (max) |
|---|---|---|---|---|---|---|
| Ollama search + Ollama fetch | 0.93 s | 1.35 s | ~22,000 tok | ~2,650 tok | 6/6 | ~1,520 tok |
| Ollama search + trafilatura | 0.85 s | 0.90 s | ~22,000 tok | ~10,200 tok | 6/6 | ~1,500 tok |
| ddgs + trafilatura | 1.46 s | 0.34 s | ~570 tok | ~9,250 tok | 5/6 | ~1,530 tok |

- **Size:** Ollama's `web_search` returns a large part of each page in its
  results (~22,000 tokens for 5 hits), so it must never reach the model raw.
  Our `web_search` cuts it to ~520 tokens with no answer lost, and
  `web_fetch` keeps about 1,000 tokens of a page at most.
- **Relevance:** Ollama's `web_fetch` returns shorter, cleaner pages. On the
  same URLs, trafilatura was blocked twice (HTTP 403 on Britannica and
  Wikipedia). The one ddgs miss had the answer in the page, past the first
  kept chunks: ranking passages by embeddings is meant to fix that case.
- **Speed:** a search plus a fetch takes 1.8 to 2.3 s (medians) on every pipeline.

**Choice:** Ollama's API first for both search and fetch, `ddgs` and a direct
download read by `trafilatura` as fallback. The fallback is about
availability, not quality: it keeps the tools working without an API key,
when free-tier quotas run out, or when the API is down.

## Model benchmark

A script, kept on the `feat/benchmarks` branch, asks a fixed set of cases to one
model at a time, through Sensai's own ReAct loop and compressor. The raw results
of each model and every table are in `benchmarks/models/results/`.
Tools are offered with their real name, description and parameters, but return
a scripted result, so the run needs no network and scores only the model.

- **Latency:** cold load, then the median of 3 warm runs, alone and after a
  ~4,300-token context.
- **Answers:** 15 questions with a single right answer, without tools.
- **Tool use:** 11 cases, scored on the first tool called, its arguments and
  the final answer; 4 of them must not call any tool.
- **Instructions:** 6 constraints on the answer (language, bullet count, JSON
  only, one word, word limit).
- **Memory:** 5 facts given early in a conversation, asked back after filler
  exchanges, first from the whole history, then from its compressed form.

**Run (2026-10-30):** RTX 4060 (8 GB), 15.3 GB of RAM, Ollama 0.32.3 with CUDA,
`num_ctx` 8192 (the context budget). Every model fit in the GPU.

| Model | Size | Generation | Cold load | Answers | Tool cases | Instructions | Memory (full / compressed) | Mean answer time |
|---|---|---|---|---|---|---|---|---|
| `qwen3:1.7b` | 1.4 GB | 161 tok/s | 2.3 s | 15/15 | 5/11 | 5/6 | 5/5 / 4/5 | 1.0 s |
| `llama3.2:3b` | 2.0 GB | 93 tok/s | 3.4 s | 15/15 | 2/11 | 2/6 | 4/5 / 5/5 | 1.7 s |
| `qwen3:4b` | 2.5 GB | 73 tok/s | 2.4 s | 15/15 | 10/11 | 1/6 | 5/5 / 5/5 | 19.4 s |
| `mistral:7b` | 4.4 GB | 53 tok/s | 4.3 s | 14/15 | 4/11 | 4/6 | 5/5 / 0/5 | 3.1 s |
| `qwen2.5:7b` | 4.7 GB | 49 tok/s | 6.1 s | 14/15 | 8/11 | 6/6 | 5/5 / 4/5 | 2.2 s |
| `gemma4:e4b` | 6.6 GB | 69 tok/s | 12.2 s | 15/15 | 11/11 | 6/6 | 5/5 / 5/5 | 2.5 s |

- **Answers do not tell models apart:** every model gets 14 or 15 of the 15
  questions right. The cases are too easy, and harder ones (multi-step
  reasoning, French phrasing) are needed to rank them on that axis.
- **Tool use does:** small models either never call a tool (`qwen3:1.7b`
  writes "call the web_search function" as text, `mistral:7b` answers from
  memory) or call one for everything (`llama3.2:3b` searches the web for a
  greeting or an addition).
- **`qwen3:4b` ignores `think: false`:** it writes its reasoning, then
  `</think>`, then the answer, all in `content`. The reasoning breaks every
  format constraint, makes answers 10 to 50 s long, and turns the 640-token
  history into a 7,054-token "summary" (114 s), larger than what it replaced.
- **`mistral:7b` loses the thread after compression:** its summary holds all
  5 facts, but the model answers the last filler question again instead of the
  one asked.
- **Context reading** goes from ~7,500 tok/s (`qwen3:1.7b`) to ~1,900 tok/s
  (`mistral:7b`): 4,300 tokens of context cost 1 to 2.5 s before the first token.

**Choice:** `gemma4:e4b` is the only model to pass every case, at a speed
close to the 4B models. It is the largest one on disk, though, and its 12 s
cold load runs past the adapter timeouts (below). `qwen2.5:7b` is the runner-up
(all instructions, 8/11 tool cases) and `qwen3:1.7b` stays the light default
for machines without a GPU.

**Issues found along the way:**

- `OllamaAdapter.load` waits 10 s: the first load of `gemma4:e4b` takes 12 to
  14 s, so selecting it fails with a read timeout. Streaming a chat waits 30 s
  between chunks, which a long context read on the CPU outlasts as well.
- `load` sends no `num_ctx`, so Ollama loads the model with its default
  context, then loads it again on the first chat, which asks for 8,192 tokens.
  Running the benchmark needed `OLLAMA_CONTEXT_LENGTH=8192` on the server.
- On NixOS, the `ollama` package is CPU-only: `ollama-cuda` is needed for the
  GPU. Its `llama-server` keeps a prompt cache of up to 8 GiB of RAM by
  default, which exhausted the memory of a 16 GB machine without swap;
  `LLAMA_ARG_CACHE_RAM` caps it (0 disables it).
