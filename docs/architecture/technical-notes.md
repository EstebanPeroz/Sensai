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
