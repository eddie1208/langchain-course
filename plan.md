# Free Tavily: SearXNG + crawl4ai search tool

Goal: a drop-in replacement for `TavilySearch` built on self-hosted SearXNG and
crawl4ai, with embedding retrieval and cross-encoder reranking, exposed to
LangChain as a tool.

## 0. Infrastructure blockers

Both containers are running. Two things will fail on the first call:

- **SearXNG blocks JSON.** `GET /search?q=test&format=json` returns 403. The
  mounted `settings.yml` is minimal and inherits defaults, which allow HTML
  only. Add `json` under `search.formats`, then restart the container.

  ```yaml
  search:
    formats:
      - html
      - json
  ```

- **crawl4ai wants auth.** Version 0.9.3 is healthy, but `/docs` and
  `/openapi.json` return `{"detail": "Authentication required"}` even though
  its `config.yml` has `security.jwt_enabled: false` and an empty `api_token`.
  Resolve the token or disable auth before writing client code.

Verify both with curl from your own shell before step 1.

Also missing from the venv: `numpy`, `flashrank`, `httpx` is present,
`langchain-community` is absent and should stay absent.

## 1. Pipeline

Five stages, each a plain function. No vector database anywhere.

| Stage  | What it does                                                        |
| ------ | ------------------------------------------------------------------- |
| Search | SearXNG JSON, 20-30 hits, dedupe by normalized URL, drop PDFs        |
| Fetch  | Top 5-8 URLs to crawl4ai markdown, parallel, hard per-URL timeout    |
| Chunk  | Split by markdown headers first, then by size, carry url + title     |
| Rank   | Embed all chunks, cosine, keep top 30, then cross-encoder to top 8   |
| Emit   | Tavily response shape so swapping is one line in `agents.py`         |

### Decisions

**Skip the vector store.** A few hundred ephemeral chunks per query, thrown
away afterwards. A numpy cosine over an array beats Chroma here and teaches
what a retriever actually does.

**Two-pass ranking, cheap then expensive.** Never rerank 400 chunks. Embedding
retrieval narrows, the cross-encoder refines.

**Reranker: FlashRank first.** CPU, milliseconds, no extra container. Graduate
to `bge-reranker-v2-m3` behind a Text Embeddings Inference container only if
quality disappoints.

**Embeddings: `OllamaEmbeddings` against `model.aihelper.best`.** Pull
`nomic-embed-text` or `bge-m3` on that host first and confirm before building
around it.

**Async everywhere.** Every stage is network IO. One shared `httpx.AsyncClient`,
and an async tool implementation so the agent can await it.

**Fetch is the latency budget.** Cap the crawl set rather than crawling all
search results.

## 2. SearXNG: raw, not the LangChain integration

`SearxSearchWrapper` lives in `langchain-community`, which is the legacy
compatibility package in LangChain 1.x. Go raw:

- It is about forty lines. One GET plus a Pydantic response model.
- The wrapper flattens results to title, link, and snippet. The raw payload
  carries `engines`, `score`, `positions`, `publishedDate`, and `category`,
  which feed ranking and freshness filtering.
- You want `time_range`, `language`, `safesearch`, `pageno`, and an engine
  allowlist varying per call. The wrapper fixes most of that at construction.

Read its source once as a reference. Do not depend on it.

## 3. Interfaces

Define domain types first, then express every interface in terms of those.

**The rule that makes this extensible: the core package imports no LangChain.**
LangChain becomes an adapter at the edge, which also lets the same pipeline be
exposed over FastAPI later.

Use `typing.Protocol` rather than abstract base classes. Structural typing means
a new implementation needs only the right method shape, with no inheritance and
no registration.

```python
class SearchProvider(Protocol):
    async def search(self, query: str, *, limit: int, **opts) -> list[SearchHit]: ...

class Crawler(Protocol):
    async def fetch(self, urls: Sequence[str]) -> list[Document]: ...

class Chunker(Protocol):
    def split(self, docs: Sequence[Document]) -> list[Chunk]: ...

class Ranker(Protocol):
    async def rank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[Scored]: ...
```

`SearchPipeline` takes all four in its constructor and does nothing else. That
constructor injection is the entire extensibility story.

Domain models as Pydantic: `SearchHit`, `Document`, `Chunk`, `Scored`,
`SearchResponse`. Interfaces speak these, never raw dicts, so swapping crawl4ai
for something else does not leak its response shape into the pipeline.

### Patterns that earn their place

Three, and no more than three.

**Chain the rankers.** A `ChainRanker` holding a list of rankers is itself a
`Ranker`. The two-stage design becomes
`ChainRanker([EmbeddingRanker(top_k=30), FlashRankRanker(top_k=8)])`.
Reordering stages, dropping the cross-encoder on a slow day, or adding a
recency booster all become list edits.

**Decorate the crawler.** Caching and retry wrap any `Crawler` and return a
`Crawler`. Write `CachedCrawler(RetryCrawler(Crawl4aiCrawler(...)))` and none
of the three knows about the others.

**Adapt at the boundary only.** One module converts `Scored` objects into the
Tavily JSON shape and into LangChain `Document` objects. Both live outside the
core.

### Two implementations per interface, from day one

Otherwise the abstraction is guesswork. Cheap second implementations:

- `TavilyProvider` wrapping the key already in `.env`, giving a quality
  baseline to measure against.
- `NullCrawler` returning snippets as documents, keeping the pipeline runnable
  when crawl4ai is down.
- `IdentityRanker` for isolating whether ranking is helping at all.

### Not doing

No config-driven plugin loader. No LLM abstraction, since LangChain already
abstracts the model. A registry pays off only once real users swap parts
without editing code.

## 4. Layout

```
tavily_free/
  models.py          SearchHit, Document, Chunk, Scored, SearchResponse
  interfaces.py      the four Protocols
  pipeline.py        SearchPipeline, constructor injection
  providers/         searxng.py, tavily.py, fake.py
  crawlers/          crawl4ai.py, null.py, cached.py, retry.py
  chunkers/          markdown.py
  rankers/           embedding.py, flashrank.py, chain.py, identity.py
  adapters/          langchain_tool.py, tavily_shape.py
```

## 5. Build order

Run each piece standalone before wrapping it.

1. Fix the SearXNG format config and the crawl4ai token. Confirm with curl.
2. Write `models.py` and `interfaces.py`. Everything else fills them in.
3. `providers/searxng.py` as a bare async function. No LangChain yet.
4. `adapters/langchain_tool.py`, then swap it into `agents.py` in place of
   `TavilySearch`. Snippets only. The agent works end to end here, which is
   the milestone that matters.
5. `crawlers/crawl4ai.py` plus chunking. Quality jumps at this step.
6. `rankers/embedding.py`, then `rankers/flashrank.py` chained on top.
7. `CachedCrawler` keyed by URL, per-stage timeouts, and graceful degradation
   to snippets when crawl4ai fails. Falling back beats raising.

LangSmith tracing is already configured in `.env`. Turn it on at step 4 and
watch where the seconds go.
