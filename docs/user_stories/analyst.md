# User Stories — Persona "Analyst"

## 1. The Persona

**The Analyst** is a data/business analyst who uses Sensai as a working
environment to run investigations: cross-referencing internal documents
(reports, specs, exports) with external data (the web), producing a reliable
synthesis, and delivering it in a usable form (report, export, dashboard).

Current pain points Sensai must address:
- Investigations span multiple sessions; the analyst doesn't want to
  re-explain the context every time.
- Internal documents are large; the analyst doesn't want to paste entire
  pages into the prompt.
- Sources must be cited, and a hallucinated figure in a deliverable is not
  acceptable.
- The data being handled is sometimes sensitive (client, financial data):
  nothing should leak to the model without control.
- The output needs to be a structured deliverable (figures, tables), not
  just free text, and sometimes a recurring report needs to be automated.

## 2. Workflow — From Situation to Feature

| # | Journey step | Concrete situation | Feature(s) triggered |
|---|---|---|---|
| 1 | Opening a session | Resuming an investigation started yesterday | Session Persistence & Profiles |
| 2 | Framing | Needing a tone/expertise adapted to the subject (e.g. finance vs. HR) | Persona · System Prompt Extension via Subject/Project |
| 3 | Internal research | Querying internal reports already ingested | RAG Basic / Advanced · File Access |
| 4 | External research | Needing a figure or a recent news item | Web Search · MCP |
| 5 | Multi-step reasoning | The question requires several chained lookups | Reasoning loops · Human in the loop |
| 6 | Long context | The investigation accumulates a lot of history/documents | Token Budgeting & Semantic Compression |
| 7 | Reliability check | Needing to trust the answer before citing it | Automated Eval & Hallucination Detection |
| 8 | Confidentiality | The data being handled is sensitive | Content & Privacy Guardrails · Adversarial testing |
| 9 | Formatting | Needing a usable result (JSON/table), not just plain text | Structured Output · Questions & forms |
| 10 | Exploring hypotheses | Testing an alternative hypothesis without losing the main thread | Branching |
| 11 | Delivery | Sharing the result with stakeholders | Export & Publication · Web UI |
| 12 | Automation | The same report needs to go out every week | Scheduling |
| 13 | Continuous improvement | Prompts get refined over time based on feedback | Prompt versioning · Semantics · Automated Prompt Optimization |

Sections 3.x below detail, feature by feature, at least two user stories
covering this workflow.

## 3. User Stories by Domain

### 3.1 Memory and Context

#### Session Persistence & Profiles

**US-01** — As an analyst, I want to close my session and resume it later with
the full conversation history intact, so that I don't have to re-explain the
context of my investigation.
Example: You close Sensai mid-investigation on Friday and reopen it Monday
— it should still remember what you were working on.

**US-02** — As an analyst, I want Sensai to remember my working profile
(preferred output format, recurring subject areas), so that repeated
instructions aren't necessary in every new session.
Example: You always want tables instead of prose; Sensai should already
know that without you repeating it every session.

#### Token Budgeting & Semantic Compression

**US-03** — As an analyst running a long investigation, I want old parts of the
conversation to be compressed instead of dropped, so that I keep the gist of
earlier findings without hitting the model's context limit.
Example: After days of back-and-forth the conversation gets very long, but
you still need to recall an early finding without it having been lost.

**US-04** — As an analyst, I want to know how much of my context budget is
being used, so that I can decide when to start a new session instead of
silently losing information.
Example: You want to see that you're close to the context limit before old
information starts silently getting lost.

### 3.2 Tools

#### MCP

**US-05** — As an analyst, I want Sensai to fetch data from an external system
(e.g. an internal API or database) as part of its reasoning, so that I
don't have to manually copy data into the chat.
Example: You need a figure from an internal system, so Sensai fetches it
directly instead of you copying it in by hand.

**US-06** — As an analyst, I want the set of external data sources available
to change depending on the investigation, so that each project can plug in
the sources relevant to it without waiting on engineering work.
Example: A new investigation needs a different data source than the last
one, and you want to plug it in without asking a developer to add it.

#### Web Search

**US-07** — As an analyst, I want to ask about a recent event or figure the
model wasn't trained on, so that I get an answer grounded in current
information instead of an outdated or fabricated one.
Example: A question needs recent information, so the assistant uses web
search to get it instead of relying on outdated training data.

**US-08** — As an analyst, I want web search failures (no network, provider
error) to degrade gracefully, so that my investigation isn't blocked by an
external outage.
Example: The network drops mid-search; you want to be told the figure
couldn't be verified instead of getting a made-up answer.

#### File Access within Permissions

**US-09** — As an analyst, I want the assistant to read, and where I allow
it write, files in an explicitly allowed project folder (e.g.
`./reports/`), so that it can use and update my local documents without me
copying content manually.
Example: Sensai reads a report from your allowed folder and writes its
summary back into that same folder when it's done.

**US-10** — As an analyst, I want an attempt to access an unauthorized path
(e.g. `../../etc/passwd`, a symlink escape) to be blocked, so that a
misbehaving or manipulated prompt can't exfiltrate data outside my project
scope.
Example: A document tries to make Sensai read a file outside your project
folder; it should refuse instead of following along.

#### Structured Output

**US-11** — As an analyst, I want to request the answer as a fixed JSON schema
(e.g. `{metric, value, source, confidence}`), so that I can feed it directly
into a spreadsheet or dashboard without manual reformatting.
Example: You need an answer as a structured record so it drops straight
into a spreadsheet instead of being retyped by hand.

**US-12** — As an analyst, I want a clear error when the model can't produce
output matching my schema, so that I know to adjust my question instead of
silently receiving malformed data.
Example: The assistant can't produce valid output for your schema; you want
a clear error, not silently broken data.

#### RAG — Basic

**US-13** — As an analyst, I want to ask questions against a folder of
internal reports I've provided to Sensai, so that the assistant answers
using our own documents instead of generic knowledge.
Example: You ask a question about a report you gave Sensai, and it answers
using that document instead of generic knowledge.

**US-14** — As an analyst, I want to update a document I've already
provided with its latest version, so that answers reflect current data
rather than a stale snapshot.
Example: You give Sensai an updated version of a document, and your next
question should use the new figures, not the old ones.

#### RAG — Advanced

**US-15** — As an analyst working across multiple projects, I want retrieval
scoped to the current subject/project, so that a query about "Project A"
doesn't surface irrelevant information from "Project B".
Example: A query about one project shouldn't return results that actually
belong to a different project.

**US-16** — As an analyst, I want the most relevant passage to surface
first when I search internal documents, so that it isn't buried under
weaker, noisier matches.
Example: The passage that actually answers your question is buried deep in
a long report; it should surface first instead of getting lost among
weaker matches.

### 3.3 Orchestration and reasoning

#### Reasoning loops

**US-17** — As an analyst, I want a multi-step question ("compare metric X
from our Q2 report to the market average, then flag the delta") to be
answered directly, with the assistant working through each step itself, so
that I don't have to break it into manual sub-questions myself.
Example: You ask a question that needs several steps, and the assistant
works through them itself instead of you asking one at a time.

**US-18** — As an analyst, I want the assistant to stop and report an
error instead of continuing forever when it can't make progress on my
request, so that a stuck investigation doesn't silently burn my time.
Example: A question has no answer anywhere in your documents; the assistant
should say so instead of searching endlessly.

#### Human in the loop

**US-19** — As an analyst, I want to be asked for confirmation before the
assistant takes an action with real consequences (e.g. writing a file,
calling a paid API), so that nothing irreversible happens without my
consent.
Example: The assistant wants to overwrite a file or call a paid API; you
confirm it before it actually happens.

**US-20** — As an analyst, I want to supply missing information the assistant
asks for mid-task (e.g. "which quarter?"), so that the investigation
continues correctly instead of the model guessing.
Example: You ask a question without specifying which period; the assistant
asks which one you mean instead of guessing.

#### Prompt versioning

**US-21** — As an analyst, I want to know exactly which persona and subject
versions were behind a given answer, so that I can reproduce or explain a
result later if challenged.
Example: You're asked to justify an answer weeks later, and you want to
know exactly which persona and project version produced it.

**US-22** — As an analyst, I want to roll back to a previous version of a
subject's or persona's prompt if a recent edit made answers worse, so that
I can recover a version I know worked without rewriting it from memory.
Example: A recent prompt edit made answers worse, and you want to go back
to the version that used to work.

#### Persona

**US-23** — As an analyst, I want to switch the assistant's persona (e.g.
"financial analyst" vs. "HR analyst" tone/expertise) to match the kind of
investigation I'm running, so that the same tool adapts to different
domains without needing engineering support.
Example: You switch from a financial task to an HR task and want the
assistant's tone and expertise to switch along with it.

**US-24** — As an analyst, I want a sensible default persona when I don't
pick one, so that the assistant behaves predictably out of the box.
Example: A new user opens Sensai without picking a persona, and it should
still behave sensibly by default.

#### Semantics

**US-25** — As an analyst, I want a loosely-worded question ("how did we do
last quarter") to be understood the same way as a precisely-worded one, so
that I don't have to phrase every query like a query language.
Example: You ask a loosely-worded question and expect it to be understood
the same way as a precise one.

**US-26** — As an analyst, I want domain-specific vocabulary (internal
acronyms, metric names) to be understood correctly, so that I don't have to
spell out our internal jargon every time.
Example: You use an internal acronym and expect the assistant to already
know what it means.

#### Automated Prompt Optimization

**US-27** — As an analyst, I want repeated failure patterns in my questions
(e.g. always needing a follow-up clarification) to inform an improved system
prompt, so that the assistant gets better at my specific use cases over
time.
Example: You keep needing to clarify the same missing detail, and the
assistant should learn to ask for it upfront next time.

**US-28** — As an analyst, I want any automatically suggested prompt change to
stay a reviewable proposal rather than being applied silently, so that the
assistant's behavior can't shift under me mid-investigation without my
knowledge.
Example: The assistant suggests a prompt change; you want to review and
approve it, not have it apply itself.

### 3.4 Evaluation

#### Automated Eval & Hallucination Detection

**US-29** — As an analyst, I want a factual claim in the assistant's answer to
be checked against real sources before I rely on it in a report, so that I
don't unknowingly cite a hallucinated figure.
Example: A draft answer includes a figure, and you want it checked against
real sources before you cite it in a report.

**US-30** — As an analyst, I want to see why a claim was flagged as possibly
hallucinated, so that I can judge for myself whether to trust it.
Example: A claim gets flagged as unverified, and you want to see why, not
just a bare warning icon.

#### Content & Privacy Guardrails

**US-31** — As an analyst handling sensitive data (client names, financial
figures), I want PII to be masked before it's sent to the model, so that
sensitive data never leaves my machine unnecessarily.
Example: You paste a client's account details into the chat to ask a
question about them, and those details should be masked before reaching
the model.

**US-32** — As an analyst, I want the assistant to refuse to output content
that violates our content policies (e.g. leaking another client's data
across sessions), so that Sensai can't be used to accidentally cross
confidentiality boundaries.
Example: A question would require pulling in another client's confidential
data, and the assistant should refuse rather than blend the two.

#### Adversarial testing

**US-33** — As an analyst pulling data from documents I didn't write myself, I
want the assistant's guardrails to have been verified against known
prompt-injection techniques (e.g. a malicious instruction hidden inside an
ingested document), so that I can trust the assistant won't be hijacked by
content it retrieves on my behalf.
Example: Before trusting the assistant with real documents, you want proof
it resists known prompt-injection tricks.

**US-34** — As an analyst, I want an instruction hidden inside a retrieved
document to be ignored by the assistant, so that untrusted document content
can't hijack the assistant's behavior on my behalf.
Example: A document contains a hidden instruction, and the assistant should
treat it as plain text instead of following it.

### 3.5 UX & lifecycle

#### Web UI

**US-35** — As an analyst, I want to use Sensai from a browser instead of the
terminal, so that I can share my screen with non-technical stakeholders
during a review.
Example: You want to show your work to a non-technical stakeholder, so you
use a browser instead of a terminal.

**US-36** — As an analyst, I want error states (model unavailable, connection
lost) to be shown clearly in the Web UI, so that I know when to retry
instead of thinking the app is frozen.
Example: The model becomes unavailable mid-session, and you want a clear
message instead of a frozen-looking screen.

#### Branching

**US-37** — As an analyst, I want to branch the conversation before exploring
an alternative hypothesis, so that I can compare two lines of investigation
without losing or contaminating the original one.
Example: You want to test an alternative hypothesis without disturbing the
investigation you already have going.

**US-38** — As an analyst, I want to see which branch a given session
descended from, so that I don't lose track of how my investigation forked
across multiple hypotheses.
Example: You find an old branched session and want to know which
investigation it originally came from.

#### Export & Publication

**US-39** — As an analyst, I want to export a finished investigation
(conversation + sources cited) to a shareable file (e.g. Markdown/PDF), so
that I can hand it to a stakeholder who doesn't have access to Sensai.
Example: An investigation is finished, and you want to hand it to someone
who doesn't have Sensai installed.

**US-40** — As an analyst, I want the export to exclude anything already
caught by the privacy guardrails, so that a shared report can't leak masked
PII that was only meant to stay internal to the assistant's reasoning.
Example: A value stayed masked throughout your session, and it should stay
masked in the exported file too.

#### Questions & forms

**US-41** — As an analyst, I want the assistant to ask me a short structured
question (e.g. pick a date range from a list) instead of a free-text
clarification when the choice is constrained, so that I answer faster and
avoid ambiguity.
Example: The assistant needs a date range from you, so it offers a short
list to pick from instead of an open-ended question.

**US-42** — As an analyst, I want to fill in several pieces of information at
once via a form (e.g. subject, date range, output format) at the start of a
recurring task, so that I don't get asked one question at a time across
multiple turns.
Example: You want to fill in several details at once instead of answering
one question per turn.

#### Scheduling

**US-43** — As an analyst, I want to schedule a recurring query (e.g. "every
Monday, summarize last week's newly added reports") so that a fresh summary
is ready for me without manually re-running it.
Example: You want a report generated automatically every Monday, without
having to ask for it yourself.

**US-44** — As an analyst, I want to list, edit, and cancel my scheduled
tasks, so that a report I no longer need doesn't keep running silently.
Example: You no longer need a report you scheduled months ago, so you find
it in a list and cancel it.

### 3.6 System Prompt Extension via Subject/Project

**US-45** — As an analyst working on a specific project, I want the
assistant's system prompt to automatically include project-specific context
(scope, vocabulary, known constraints) when I select that project, so that I
don't have to repeat project background in every session.
Example: Switching to a project should bring its background and vocabulary
with it, without you repeating it yourself.

**US-46** — As an analyst starting a brand-new type of investigation, I want
to set up a new subject/project myself, so that onboarding a new analysis
domain doesn't require waiting on a developer.
Example: You want to set up a brand-new type of project yourself today,
without waiting on a developer.
