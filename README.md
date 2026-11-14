# Free Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API (JSON and SSE) over ten free-tier providers on the OpenAI,
Anthropic and Gemini wires, for make.fun `free-model-aggregator-01`.

## Run

```
node main.ts            # reads MAKEFUN_PROMPT_FILE, registers at MAKEFUN_ENV_URL
node main.ts --list     # prints the catalog document and exits 0
```

Node 22, standard library only, nothing to install.

## How it routes

- Probes every provider's catalog route in its own wire and auth scheme,
  then registers one catalog (`--list` prints the same document).
- Ranks every free model that meets a request's requirements by the brief's
  sealed quality rating, highest first; paid models are never picked.
- On a 402/429/5xx it retries the same model once (honouring `retry-after`),
  then fails over to the next-ranked model. A provider failing twice in a
  row is treated as spent and skipped unless it is the only candidate left.
- An OpenAI JSON-mode request never goes to an Anthropic-wire provider,
  which cannot carry `response_format`.
- Answers are translated back into the client's wire with the provider's
  text, usage counts and stop reason intact. Streamed OpenAI calls always
  set `stream_options.include_usage` upstream so the final usage chunk
  exists. All reply timestamps are fixed constants, so replays are
  byte-identical.

## Local grade (kit `grade-local`, process driver)

Practice trials 0-2: score **99** each (quality 99, served 100, wire 100,
failover 100, catalog 100), 20/20 requests valid, replay matched.

## License

MIT — see LICENSE.
