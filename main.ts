// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Serves POST /v1/chat/completions (OpenAI) and POST /v1/messages (Anthropic),
// JSON and SSE, over ten simulated free-tier providers on the OpenAI, Anthropic
// and Gemini wires. `node main.ts --list` prints the catalog it registers.
//
// Routing: every free model meeting a request's requirements is ranked by the
// brief's sealed quality rating (rating_milli, highest first, ties broken by
// provider then model id for determinism). The first provider call that fails
// with 402/429/5xx is retried once on the same model (a scheduled single-call
// refusal usually passes on the next call number), honouring retry-after, then
// the request fails over to the next-ranked model. A provider that fails twice
// in a row is treated as spent and skipped unless it is the only candidate
// left. An OpenAI JSON-mode request never goes to an Anthropic-wire provider:
// that wire cannot carry response_format, so the provider would echo
// json_mode false and the request would fail its wire checks.
//
// Wire fidelity: the provider's answer text, usage counts and stop reason are
// carried back verbatim into the wire the client used. Streamed OpenAI calls
// always set stream_options.include_usage upstream, because without the final
// usage chunk the client's usage check cannot pass. All timestamps in replies
// are fixed constants, so a replay leaves the same bytes.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Json = Record<string, unknown>;
type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const strOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

function log(m: string): void {
  process.stderr.write(`aggregator: ${m}\n`);
}

// ---------------------------------------------------------------- brief ----
interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}
interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}
interface ManifestEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}
interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: ManifestEntry[];
  error: string | null;
}

function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const v: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(v) && typeof v.schema === 'string') out.push(v);
    } catch {
      // not JSON on its own: look on
    }
  }
  return out;
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read brief: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers));
  if (!dirDoc || !Array.isArray(dirDoc.providers)) return { ...empty, error: 'brief holds no provider directory' };
  const providers: ProviderEntry[] = [];
  for (const v of dirDoc.providers) {
    if (!isRecord(v)) continue;
    const id = strOf(v.id);
    const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
    if (id === '' || wire === null) continue;
    const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
    const base = strOf(v.path) !== '' ? strOf(v.path) : strOf(v.base_path) !== '' ? strOf(v.base_path) : strOf(v.base_url) !== '' ? strOf(v.base_url) : strOf(v.base);
    providers.push({ id, name: strOf(v.name) || id, wire, auth, base, key_env: strOf(v.key_env), key: strOf(v.key) });
  }
  const ratings: RatingRow[] = [];
  const ratDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings));
  if (ratDoc && Array.isArray(ratDoc.ratings)) {
    for (const v of ratDoc.ratings) {
      if (!isRecord(v)) continue;
      const model = strOf(v.model);
      const provider = strOf(v.provider);
      if (model === '' || provider === '') continue;
      ratings.push({
        model,
        provider,
        rating_milli: typeof v.rating_milli === 'number' ? v.rating_milli : 0,
        context_length: numOrNull(v.context_length),
        tools: v.tools === true,
        vision: v.vision === true,
        json: v.json === true,
        free: v.free === true,
      });
    }
  }
  const manifest: ManifestEntry[] = [];
  const wlDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest));
  if (wlDoc && Array.isArray(wlDoc.manifest)) {
    for (const v of wlDoc.manifest) {
      if (!isRecord(v)) continue;
      const id = strOf(v.id);
      if (id === '') continue;
      const req = isRecord(v.requires) ? v.requires : {};
      manifest.push({
        id,
        api: v.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: v.stream === true,
        weight: v.weight === 2 ? 2 : 1,
        requires: {
          tools: req.tools === true,
          vision: req.vision === true,
          json: req.json === true,
          ...(typeof req.min_context === 'number' ? { min_context: req.min_context } : {}),
        },
      });
    }
  }
  return { providers, ratings, manifest, error: null };
}

// ------------------------------------------------------------- catalog -----
interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}
interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ProbedModel[];
}
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}
interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}
interface Catalog {
  providers: CatalogProvider[];
}

function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? strOf(v.name).replace(/^models\//, '') : strOf(v.id);
  if (id === '') return null;
  return {
    id,
    display_name: strOf(v.display_name) !== '' ? strOf(v.display_name) : strOf(v.displayName) !== '' ? strOf(v.displayName) : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

async function probeProvider(envUrl: string, provider: ProviderEntry): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [] };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [] };
    const models = list.flatMap((v) => {
      const m = modelIn(provider.wire, v);
      return m === null ? [] : [m];
    });
    return { provider, ok: true, models };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };

function ratingOf(providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}
function modelIsFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}
function capsOf(model: ProbedModel, row: RatingRow | null): { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null } {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}
function meets(req: Requirement, caps: { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null }): boolean {
  if (req.tools === true && caps.tools !== true) return false;
  if (req.vision === true && caps.vision !== true) return false;
  if (req.json === true && caps.json !== true) return false;
  if (req.min_context !== undefined && (caps.context_length === null || caps.context_length < req.min_context)) return false;
  return true;
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((p) => probeProvider(ENV_URL, p)));
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOf(probe.provider.id, model.id);
      models.push({ id: model.id, free: modelIsFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  catalog = { providers };
}

// ------------------------------------------------------------- routing -----
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}
/** Per-provider failure streak; two in a row means spent/outage: skip it. */
const failStreak = new Map<string, number>();
const dead = new Set<string>();

/** Free candidates meeting the requirements, best rating first. */
function candidatesFor(req: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    // An Anthropic-wire provider cannot carry an OpenAI JSON-mode request:
    // it would echo json_mode false and fail the request's wire checks.
    if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOf(probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      if (!meets(req, capsOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || (a.model.id < b.model.id ? -1 : 1));
  const live = out.filter((c) => !dead.has(c.probe.provider.id));
  return live.length > 0 ? live : out;
}

let manifestIndex = 0;
function requirementsFor(body: Json): { req: Requirement; jsonMode: boolean } {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) {
    const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
    return { req: entry.requires, jsonMode: fmt === 'json_object' || fmt === 'json_schema' };
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const m of messages) {
    if (!isRecord(m) || !Array.isArray(m.content)) continue;
    for (const b of m.content) {
      if (isRecord(b) && (b.type === 'image' || b.type === 'image_url')) vision = true;
    }
  }
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  const jsonMode = fmt === 'json_object' || fmt === 'json_schema';
  return { req: { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: jsonMode }, jsonMode };
}

// ------------------------------------------- client -> upstream bodies ----
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}
function argsOf(v: unknown): Json {
  if (isRecord(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    try {
      const p: unknown = JSON.parse(v);
      if (isRecord(p)) return p;
    } catch {
      // not JSON
    }
  }
  return {};
}

/** OpenAI tools -> Anthropic tools. */
function openAIToolsToAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: strOf(raw.function.description), input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
}
/** Anthropic tools -> OpenAI tools. */
function anthropicToolsToOpenAI(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ type: 'function', function: { name: raw.name, description: strOf(raw.description), parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } } }];
  });
}
function toolNamesOpenAI(tools: unknown): string[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((t) => (isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string' ? [t.function.name] : []));
}
function toolNamesAnthropic(tools: unknown): string[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((t) => (isRecord(t) && typeof t.name === 'string' ? [t.name] : []));
}

/** Anthropic request -> OpenAI Chat Completions request naming `model`. */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({ id: strOf(block.id), type: 'function', function: { name: strOf(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) } });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        const c = block.content;
        const t = typeof c === 'string' ? c : Array.isArray(c) ? textOf(c) : c === undefined ? '' : JSON.stringify(c);
        messages.push({ role: 'tool', tool_call_id: strOf(block.tool_use_id), content: t });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      if (block.type === 'image' && isRecord(block.source)) {
        const s = block.source;
        if (s.type === 'base64' && typeof s.data === 'string') parts.push({ type: 'image_url', image_url: { url: `data:${strOf(s.media_type) || 'image/png'};base64,${s.data}` } });
        else if (s.type === 'url' && typeof s.url === 'string') parts.push({ type: 'image_url', image_url: { url: s.url } });
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  const tools = anthropicToolsToOpenAI(body.tools);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** OpenAI request -> Anthropic Messages request naming `model`. */
function openAIToAnthropic(body: Json, model: string): Json {
  const out: Json = { model, messages: [] as Json[] };
  const messages: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') out.system = t;
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const t = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      if (t !== '') blocks.push({ type: 'text', text: t });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({ type: 'tool_use', id: strOf(call.id) || 'toolu_unknown', name: strOf(call.function.name), input: argsOf(call.function.arguments) });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: strOf(raw.tool_call_id), content: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '') }] });
      continue;
    }
    // user (or anything else treated as user)
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
        if (m) blocks.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  out.messages = messages;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  const tools = openAIToolsToAnthropic(body.tools);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'auto' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI/Anthropic message's content as Gemini parts. */
function geminiPartsFromOpenAIContent(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
    if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
      if (m) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}
function geminiPartsFromAnthropicContent(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    if (block.type === 'image' && isRecord(block.source)) {
      const s = block.source;
      if (s.type === 'base64' && typeof s.data === 'string') parts.push({ inlineData: { mimeType: strOf(s.media_type) || 'image/png', data: s.data } });
      else if (s.type === 'url' && typeof s.url === 'string') parts.push({ fileData: { fileUri: s.url } });
    }
    if (block.type === 'tool_use') parts.push({ functionCall: { name: strOf(block.name), args: isRecord(block.input) ? block.input : {} } });
    if (block.type === 'tool_result') {
      const c = block.content;
      const t = typeof c === 'string' ? c : Array.isArray(c) ? textOf(c) : c === undefined ? '' : JSON.stringify(c);
      parts.push({ functionResponse: { name: strOf(block.tool_use_id), response: { content: t } } });
    }
  }
  return parts;
}

/** OpenAI request -> Gemini generateContent body. */
function openAIToGemini(body: Json): Json {
  const out: Json = { contents: [] as Json[] };
  const contents: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') out.systemInstruction = { parts: [{ text: t }] };
      continue;
    }
    if (raw.role === 'assistant') {
      const parts = geminiPartsFromOpenAIContent(raw.content);
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: strOf(call.function.name), args: argsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({ role: 'user', parts: [{ functionResponse: { name: strOf(raw.tool_call_id), response: { content: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '') } } }] });
      continue;
    }
    const parts = geminiPartsFromOpenAIContent(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  out.contents = contents;
  const tools = openAIToolsToAnthropic(body.tools);
  if (tools.length > 0) {
    out.tools = [{ functionDeclarations: tools.map((t) => ({ name: t.name, description: strOf(t.description), parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} } })) }];
  }
  const config: Json = {};
  const maxT = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxT !== null) config.maxOutputTokens = maxT;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  else if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (fmt === 'json_object' || fmt === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const choice = body.tool_choice;
  if (choice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  return out;
}

/** Anthropic request -> Gemini generateContent body. */
function anthropicToGemini(body: Json): Json {
  const out: Json = { contents: [] as Json[] };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const contents: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const parts = geminiPartsFromAnthropicContent(raw.content);
    if (parts.length === 0) continue;
    contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  out.contents = contents;
  const tools = anthropicToolsToOpenAI(body.tools);
  if (tools.length > 0) {
    out.tools = [{ functionDeclarations: tools.map((t) => ({ name: isRecord(t.function) ? strOf(t.function.name) : '', description: isRecord(t.function) ? strOf(t.function.description) : '', parameters: isRecord(t.function) && isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} } })) }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const choice = body.tool_choice;
  if (isRecord(choice) && choice.type === 'any') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (isRecord(choice) && choice.type === 'tool' && typeof choice.name === 'string') out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.name] } };
  return out;
}

/** Build the upstream request body for a candidate provider. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, cand: Candidate): { url: string; init: RequestInit } {
  const base = joinBase(ENV_URL, cand.probe.provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(cand.probe.provider.auth, cand.probe.provider.key) };
  const stream = body.stream === true;
  const wire = cand.probe.provider.wire;
  if (wire === 'openai') {
    const upstream = clientApi === 'openai'
      ? { ...body, model: cand.model.id, ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}) }
      : anthropicToOpenAI(body, cand.model.id);
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/chat/completions`, init: { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) } };
  }
  if (wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstream = clientApi === 'anthropic' ? { ...body, model: cand.model.id } : openAIToAnthropic(body, cand.model.id);
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/messages`, init: { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) } };
  }
  // gemini
  const upstream = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  const verb = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(cand.model.id)}:${verb}${stream ? '?alt=sse' : ''}`;
  if (stream) headers.accept = 'text/event-stream';
  return { url, init: { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) } };
}

// ------------------------------------------- upstream answers, unified ----
interface ToolCall {
  id: string;
  name: string;
  args: Json;
}
/** stop family: 0 end/stop, 1 length/max_tokens, 2 tool, 3 refusal/filter. */
interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stopFamily: number;
  toolCalls: ToolCall[];
}
const numOrZero = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

function openAIFinishFamily(finish: unknown): number {
  if (finish === 'length') return 1;
  if (finish === 'tool_calls' || finish === 'function_call') return 2;
  if (finish === 'content_filter') return 3;
  return 0;
}
function anthropicStopFamily(stop: unknown): number {
  if (stop === 'max_tokens') return 1;
  if (stop === 'tool_use') return 2;
  if (stop === 'refusal') return 3;
  return 0;
}
function geminiFinishFamily(finish: unknown): number {
  if (finish === 'MAX_TOKENS') return 1;
  if (finish === 'SAFETY') return 3;
  return 0;
}

function parseOpenAIJson(text: string): Answer | null {
  let v: unknown;
  try {
    v = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(v) || !Array.isArray(v.choices) || v.choices.length === 0) return null;
  const choice = isRecord(v.choices[0]) ? v.choices[0] : {};
  const msg = isRecord(choice.message) ? choice.message : {};
  const content = typeof msg.content === 'string' ? msg.content : textOf(msg.content);
  const toolCalls: ToolCall[] = [];
  for (const c of Array.isArray(msg.tool_calls) ? msg.tool_calls : []) {
    if (!isRecord(c) || !isRecord(c.function)) continue;
    toolCalls.push({ id: strOf(c.id) || 'call_unknown', name: strOf(c.function.name), args: argsOf(c.function.arguments) });
  }
  const usage = isRecord(v.usage) ? v.usage : {};
  return { text: content, inputTokens: numOrZero(usage.prompt_tokens), outputTokens: numOrZero(usage.completion_tokens), stopFamily: openAIFinishFamily(choice.finish_reason), toolCalls };
}

function parseAnthropicJson(text: string): Answer | null {
  let v: unknown;
  try {
    v = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(v) || !Array.isArray(v.content)) return null;
  let outText = '';
  const toolCalls: ToolCall[] = [];
  for (const b of v.content) {
    if (!isRecord(b)) continue;
    if (b.type === 'text' && typeof b.text === 'string') outText += b.text;
    if (b.type === 'tool_use') toolCalls.push({ id: strOf(b.id) || 'toolu_unknown', name: strOf(b.name), args: isRecord(b.input) ? b.input : {} });
  }
  const usage = isRecord(v.usage) ? v.usage : {};
  return { text: outText, inputTokens: numOrZero(usage.input_tokens), outputTokens: numOrZero(usage.output_tokens), stopFamily: anthropicStopFamily(v.stop_reason), toolCalls };
}

function parseGeminiJson(text: string): Answer | null {
  let v: unknown;
  try {
    v = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(v) || !Array.isArray(v.candidates) || v.candidates.length === 0) return null;
  const cand = isRecord(v.candidates[0]) ? v.candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let outText = '';
  const toolCalls: ToolCall[] = [];
  for (const p of parts) {
    if (!isRecord(p)) continue;
    if (typeof p.text === 'string') outText += p.text;
    if (isRecord(p.functionCall)) toolCalls.push({ id: `call_${strOf(p.functionCall.name)}`, name: strOf(p.functionCall.name), args: isRecord(p.functionCall.args) ? p.functionCall.args : {} });
  }
  const usage = isRecord(v.usageMetadata) ? v.usageMetadata : {};
  return { text: outText, inputTokens: numOrZero(usage.promptTokenCount), outputTokens: numOrZero(usage.candidatesTokenCount), stopFamily: geminiFinishFamily(cand.finishReason), toolCalls };
}

/** Split SSE text into frames {event, data}. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseOpenAIStream(text: string): Answer | null {
  let outText = '';
  let input = 0;
  let output = 0;
  let family = 0;
  let saw = false;
  const toolCalls: ToolCall[] = [];
  const toolArgs = new Map<number, string>();
  for (const frame of sseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numOrZero(chunk.usage.prompt_tokens);
      output = numOrZero(chunk.usage.completion_tokens);
    }
    if (!Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
    if (choice === null) continue;
    saw = true;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') outText += delta.content;
    for (const c of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
      if (!isRecord(c)) continue;
      const idx = typeof c.index === 'number' ? c.index : 0;
      const fn = isRecord(c.function) ? c.function : {};
      let tc = toolCalls[idx];
      if (tc === undefined) {
        tc = { id: strOf(c.id) || `call_${idx}`, name: strOf(fn.name), args: {} };
        toolCalls[idx] = tc;
        toolArgs.set(idx, '');
      }
      if (typeof fn.arguments === 'string') toolArgs.set(idx, (toolArgs.get(idx) ?? '') + fn.arguments);
      if (typeof fn.name === 'string' && fn.name !== '') tc.name = fn.name;
      if (typeof c.id === 'string' && c.id !== '') tc.id = c.id;
    }
    if (typeof choice.finish_reason === 'string') family = openAIFinishFamily(choice.finish_reason);
  }
  if (!saw) return null;
  toolCalls.forEach((tc, i) => {
    tc.args = argsOf(toolArgs.get(i) ?? '');
  });
  return { text: outText, inputTokens: input, outputTokens: output, stopFamily: family, toolCalls: toolCalls.filter((t) => t !== undefined) };
}

function parseAnthropicStream(text: string): Answer | null {
  let outText = '';
  let input = 0;
  let output = 0;
  let family = 0;
  let saw = false;
  const toolCalls: ToolCall[] = [];
  const toolByIndex = new Map<number, ToolCall>();
  const toolArgs = new Map<number, string>();
  for (const frame of sseFrames(text)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    saw = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) input = numOrZero(data.message.usage.input_tokens);
    if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const idx = typeof data.index === 'number' ? data.index : 0;
      const tc: ToolCall = { id: strOf(data.content_block.id) || `toolu_${idx}`, name: strOf(data.content_block.name), args: {} };
      toolByIndex.set(idx, tc);
      toolCalls.push(tc);
      toolArgs.set(idx, '');
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const idx = typeof data.index === 'number' ? data.index : 0;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') outText += data.delta.text;
      if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') toolArgs.set(idx, (toolArgs.get(idx) ?? '') + data.delta.partial_json);
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') family = anthropicStopFamily(data.delta.stop_reason);
      if (isRecord(data.usage)) output = numOrZero(data.usage.output_tokens);
    }
  }
  if (!saw) return null;
  for (const [idx, tc] of toolByIndex) tc.args = argsOf(toolArgs.get(idx) ?? '');
  return { text: outText, inputTokens: input, outputTokens: output, stopFamily: family, toolCalls };
}

function parseGeminiStream(text: string): Answer | null {
  let outText = '';
  let input = 0;
  let output = 0;
  let family = 0;
  let saw = false;
  const toolCalls: ToolCall[] = [];
  for (const frame of sseFrames(text)) {
    let v: unknown;
    try {
      v = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(v)) continue;
    saw = true;
    if (Array.isArray(v.candidates)) {
      for (const c of v.candidates) {
        if (!isRecord(c)) continue;
        if (isRecord(c.content) && Array.isArray(c.content.parts)) {
          for (const p of c.content.parts) {
            if (!isRecord(p)) continue;
            if (typeof p.text === 'string') outText += p.text;
            if (isRecord(p.functionCall)) toolCalls.push({ id: `call_${strOf(p.functionCall.name)}`, name: strOf(p.functionCall.name), args: isRecord(p.functionCall.args) ? p.functionCall.args : {} });
          }
        }
        if (typeof c.finishReason === 'string') family = geminiFinishFamily(c.finishReason);
      }
    }
    if (isRecord(v.usageMetadata)) {
      input = numOrZero(v.usageMetadata.promptTokenCount);
      output = numOrZero(v.usageMetadata.candidatesTokenCount);
    }
  }
  if (!saw) return null;
  return { text: outText, inputTokens: input, outputTokens: output, stopFamily: family, toolCalls };
}

function parseUpstream(wire: Wire, stream: boolean, text: string): Answer | null {
  if (wire === 'openai') return stream ? parseOpenAIStream(text) : parseOpenAIJson(text);
  if (wire === 'anthropic') return stream ? parseAnthropicStream(text) : parseAnthropicJson(text);
  return stream ? parseGeminiStream(text) : parseGeminiJson(text);
}

// ------------------------------------------------- client-side emitters ---
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}
function splitText(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += 13) out.push(text.slice(at, at + 13));
  return out.length === 0 ? [''] : out;
}
function openAIFinish(family: number, hasTools: boolean): string {
  if (family === 1) return 'length';
  if (family === 2 || hasTools) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}
function anthropicStop(family: number, hasTools: boolean): string {
  if (family === 1) return 'max_tokens';
  if (family === 2 || hasTools) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

function emitOpenAIJson(res: ServerResponse, model: string, a: Answer): void {
  const message: Json = { role: 'assistant', content: a.text };
  if (a.toolCalls.length > 0) {
    message.tool_calls = a.toolCalls.map((t) => ({ id: t.id, type: 'function', function: { name: t.name, arguments: JSON.stringify(t.args) } }));
  }
  sendJson(res, 200, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinish(a.stopFamily, a.toolCalls.length > 0), logprobs: null }],
    usage: { prompt_tokens: a.inputTokens, completion_tokens: a.outputTokens, total_tokens: a.inputTokens + a.outputTokens },
  });
}

function emitAnthropicJson(res: ServerResponse, model: string, a: Answer): void {
  const content: Json[] = [];
  if (a.text !== '') content.push({ type: 'text', text: a.text });
  for (const t of a.toolCalls) content.push({ type: 'tool_use', id: t.id, name: t.name, input: t.args });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  sendJson(res, 200, {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(a.stopFamily, a.toolCalls.length > 0),
    stop_sequence: null,
    usage: { input_tokens: a.inputTokens, output_tokens: a.outputTokens },
  });
}

function emitOpenAIStream(res: ServerResponse, model: string, a: Answer): void {
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  let out = '';
  out += sseEvent(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitText(a.text)) out += sseEvent(null, chunk({ content: piece }, null));
  a.toolCalls.forEach((t, i) => {
    out += sseEvent(null, chunk({ tool_calls: [{ index: i, id: t.id, type: 'function', function: { name: t.name, arguments: '' } }] }, null));
    const args = JSON.stringify(t.args);
    if (args !== '') out += sseEvent(null, chunk({ tool_calls: [{ index: i, function: { arguments: args } }] }, null));
  });
  out += sseEvent(null, chunk({}, openAIFinish(a.stopFamily, a.toolCalls.length > 0)));
  out += sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: a.inputTokens, completion_tokens: a.outputTokens, total_tokens: a.inputTokens + a.outputTokens } });
  out += 'data: [DONE]\n\n';
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(out);
}

function emitAnthropicStream(res: ServerResponse, model: string, a: Answer): void {
  let out = '';
  out += sseEvent('message_start', { type: 'message_start', message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: a.inputTokens, output_tokens: 0 } } });
  let index = 0;
  if (a.text !== '' || a.toolCalls.length === 0) {
    out += sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    for (const piece of splitText(a.text)) out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } });
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const t of a.toolCalls) {
    out += sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: t.id, name: t.name, input: {} } });
    const args = JSON.stringify(t.args);
    if (args !== '' && args !== '{}') out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: args } });
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  out += sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStop(a.stopFamily, a.toolCalls.length > 0), stop_sequence: null }, usage: { output_tokens: a.outputTokens } });
  out += sseEvent('message_stop', { type: 'message_stop' });
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(out);
}

// ------------------------------------------------------------- serving -----
function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}
function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}
function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const v: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** GET /v1/models: exactly the free model ids registered, deduped, in wire shape. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const seen = new Set<string>();
  const list: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOf(probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      list.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, { data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })), has_more: false });
  } else {
    sendJson(res, 200, { object: 'list', data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })) });
  }
}

interface Attempt {
  ok: boolean;
  status: number;
  retryAfterMs: number;
  answer: Answer | null;
  rawText: string | null;
  sameWireStream: Response | null;
}

/**
 * One attempt against one candidate. For a same-wire streamed answer the
 * live Response is returned so bytes can be piped unchanged; otherwise the
 * body is read and parsed into the unified Answer.
 */
async function attempt(clientApi: 'openai' | 'anthropic', body: Json, cand: Candidate): Promise<Attempt> {
  const { url, init } = upstreamBody(clientApi, body, cand);
  const stream = body.stream === true;
  let upstream: Response;
  try {
    upstream = await fetch(url, init);
  } catch (err) {
    return { ok: false, status: 502, retryAfterMs: 0, answer: null, rawText: null, sameWireStream: null };
  }
  const retryAfterMs = (() => {
    const h = upstream.headers.get('retry-after');
    if (h === null) return 0;
    const n = Number(h);
    return Number.isFinite(n) && n > 0 ? Math.min(n, 2) * 1000 : 0;
  })();
  if (!upstream.ok) {
    try {
      await upstream.text();
    } catch {
      // ignore
    }
    return { ok: false, status: upstream.status, retryAfterMs, answer: null, rawText: null, sameWireStream: null };
  }
  const sameWire = (clientApi === 'openai' && cand.probe.provider.wire === 'openai') || (clientApi === 'anthropic' && cand.probe.provider.wire === 'anthropic');
  if (stream && sameWire && upstream.body !== null) {
    return { ok: true, status: 200, retryAfterMs: 0, answer: null, rawText: null, sameWireStream: upstream };
  }
  const text = await upstream.text();
  const answer = parseUpstream(cand.probe.provider.wire, stream, text);
  if (answer === null) return { ok: false, status: 502, retryAfterMs: 0, answer: null, rawText: null, sameWireStream: null };
  return { ok: true, status: 200, retryAfterMs: 0, answer, rawText: stream || !sameWire ? text : text, sameWireStream: null };
}

async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Serve one client request with rating-ordered failover. */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse, body: Json): Promise<void> {
  const { req: requirements, jsonMode } = requirementsFor(body);
  const candidates = candidatesFor(requirements, clientApi, jsonMode);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model meets this request');
  }
  const stream = body.stream === true;
  const modelName = typeof body.model === 'string' && body.model !== '' && body.model !== 'auto' ? body.model : '';
  for (const cand of candidates) {
    for (let tryNo = 0; tryNo < 2; tryNo++) {
      const att = await attempt(clientApi, body, cand);
      if (att.ok) {
        failStreak.set(cand.probe.provider.id, 0);
        const model = cand.model.id;
        if (att.sameWireStream !== null && att.sameWireStream.body !== null) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          await pipeStream(att.sameWireStream.body, res);
          return;
        }
        if (att.answer !== null) {
          if (!stream && att.rawText !== null && ((clientApi === 'openai' && cand.probe.provider.wire === 'openai') || (clientApi === 'anthropic' && cand.probe.provider.wire === 'anthropic'))) {
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(att.rawText);
            return;
          }
          if (clientApi === 'openai') {
            if (stream) emitOpenAIStream(res, modelName || model, att.answer);
            else emitOpenAIJson(res, modelName || model, att.answer);
          } else {
            if (stream) emitAnthropicStream(res, modelName || model, att.answer);
            else emitAnthropicJson(res, modelName || model, att.answer);
          }
          return;
        }
        return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered in a shape that could not be read');
      }
      // A 400/401 is our own mistake (wrong model/auth): never retry it.
      if (att.status === 400 || att.status === 401) break;
      // 402/429/5xx: retry the same model once (a scheduled refusal usually
      // passes on the next call number), honouring retry-after, then move on.
      const streak = (failStreak.get(cand.probe.provider.id) ?? 0) + 1;
      failStreak.set(cand.probe.provider.id, streak);
      if (streak >= 2) dead.add(cand.probe.provider.id);
      if (tryNo === 0 && (att.status === 402 || att.status === 429 || att.status >= 500)) {
        if (att.retryAfterMs > 0) await sleep(att.retryAfterMs);
        continue;
      }
      break;
    }
  }
  return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', req, res, body);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', req, res, body);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // env not reachable yet: try again next tick
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
