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
This is the fallback path `tools/registry.py` would need to branch on
per-model.

## The reasoning loop (ReAct)

ReAct = reflection + action, combined. After a user prompt, the model
enters:

```
reflect -> act -> observe result -> reflect -> act -> observe result -> ...
```

The hard part, called out explicitly in the notes, is **detecting when
reflection has reached the expected result** — i.e. the loop's termination
condition. This is the open design problem for
`core/reasoning_loop.py`.

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
