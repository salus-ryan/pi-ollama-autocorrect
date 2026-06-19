import {
  CustomEditor,
  type ExtensionAPI,
  type ExtensionContext,
  type KeybindingsManager,
} from "@earendil-works/pi-coding-agent";
import type { EditorTheme, TUI } from "@earendil-works/pi-tui";
import { CURSOR_MARKER, Key, matchesKey, wrapTextWithAnsi } from "@earendil-works/pi-tui";

const DEFAULT_MODEL = "llama3.2:latest";
const OLLAMA_URL = "http://127.0.0.1:11434/api/generate";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";
const OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini";
const GHOST_DEBOUNCE_MS = 900;
const GHOST_MIN_CHARS = 8;
const PREDICTION_MIN_CHARS = 16;

// Keep live typing to one Ollama request at a time. On-device Ollama usually queues
// generations, so racing autocorrect + prediction while typing can hide prediction.
// Autocorrect remains available via Tab after manual command/shortcut and /autocorrect.
const GHOST_MODEL = "llama3.2:latest";
const PREDICTION_MODEL = "llama3.2:latest";
const FULL_RACE_MODELS = ["qwen2.5-coder:1.5b", "llama3.2:latest", "mistral:latest", "phi:latest"];
const PREDICTION_RACE_MODELS = ["ollama:llama3.2:latest", `openrouter:${OPENROUTER_DEFAULT_MODEL}`];

type RaceCandidate = {
  model: string;
  text?: string;
  ms: number;
  error?: string;
};

type RaceResult = {
  input: string;
  winner: string;
  agreement: number;
  total: number;
  candidates: RaceCandidate[];
};

function normalizeForAgreement(text: string): string {
  return text.trim().replace(/\s+/g, " ").toLowerCase();
}

function getEnv(name: string): string | undefined {
  return (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.[name];
}

function cleanContinuation(original: string, raw: string): string {
  const withoutQuotes = raw.trim().replace(/^['"“”]+|['"“”]+$/g, "");
  return withoutQuotes.startsWith(original) ? withoutQuotes.slice(original.length).trim() : withoutQuotes;
}

async function askOllama(original: string, model = DEFAULT_MODEL, signal?: AbortSignal): Promise<string> {
  const response = await fetch(OLLAMA_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    signal,
    body: JSON.stringify({
      model,
      stream: false,
      prompt: [
        "You are an autocorrect engine for a coding-agent input textbox.",
        "Correct spelling, punctuation, capitalization, and obvious grammar only.",
        "Preserve the user's meaning, tone, formatting, line breaks, code, shell commands, file paths, URLs, @file references, slash commands, and quoted text.",
        "Do not answer the prompt. Do not add commentary. Return only the corrected text.",
        "",
        "Text:",
        original,
      ].join("\n"),
      options: {
        temperature: 0,
        num_predict: Math.max(128, Math.ceil(original.length * 1.5)),
      },
    }),
  });

  if (!response.ok) {
    throw new Error(`Ollama HTTP ${response.status}: ${await response.text()}`);
  }

  const data = (await response.json()) as { response?: string };
  return data.response?.trim() ?? "";
}

async function predictNext(original: string, model = PREDICTION_MODEL, signal?: AbortSignal): Promise<string> {
  const response = await fetch(OLLAMA_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    signal,
    body: JSON.stringify({
      model,
      stream: false,
      prompt: [
        "You are a typing continuation engine for a coding-agent input textbox.",
        "Predict only the next short phrase or sentence the user is likely to type.",
        "Preserve the user's tone and context. Do not answer the user. Do not explain.",
        "Return only the continuation text, not the original text.",
        "Always return a plausible continuation, even if you are uncertain.",
        "",
        "Text so far:",
        original,
      ].join("\n"),
      options: {
        temperature: 0.2,
        num_predict: 48,
      },
    }),
  });

  if (!response.ok) {
    throw new Error(`Ollama HTTP ${response.status}: ${await response.text()}`);
  }

  const data = (await response.json()) as { response?: string };
  return cleanContinuation(original, data.response ?? "");
}

async function predictNextOpenRouter(
  original: string,
  model = OPENROUTER_DEFAULT_MODEL,
  signal?: AbortSignal,
): Promise<string> {
  const apiKey = getEnv("OPENROUTER_API_KEY");
  if (!apiKey) throw new Error("OPENROUTER_API_KEY is not set");

  const response = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: {
      "authorization": `Bearer ${apiKey}`,
      "content-type": "application/json",
      "http-referer": "https://github.com/salus-ryan/pi-ollama-autocorrect",
      "x-title": "pi-ollama-autocorrect",
    },
    signal,
    body: JSON.stringify({
      model,
      temperature: 0.2,
      max_tokens: 48,
      messages: [
        {
          role: "system",
          content: [
            "You are a typing continuation engine for a coding-agent input textbox.",
            "Predict only the next short phrase or sentence the user is likely to type.",
            "Preserve the user's tone and context. Do not answer the user. Do not explain.",
            "Return only the continuation text, not the original text.",
            "Always return a plausible continuation, even if you are uncertain.",
          ].join(" "),
        },
        { role: "user", content: `Text so far:\n${original}` },
      ],
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenRouter HTTP ${response.status}: ${await response.text()}`);
  }

  const data = (await response.json()) as { choices?: Array<{ message?: { content?: string } }> };
  return cleanContinuation(original, data.choices?.[0]?.message?.content ?? "");
}

async function predictNextWithSpec(original: string, spec: string, signal?: AbortSignal): Promise<string> {
  if (spec.startsWith("openrouter:")) return predictNextOpenRouter(original, spec.slice("openrouter:".length), signal);
  if (spec.startsWith("ollama:")) return predictNext(original, spec.slice("ollama:".length), signal);
  return predictNext(original, spec, signal);
}

function chooseRaceWinner(input: string, candidates: RaceCandidate[]): RaceResult {
  const counts = new Map<string, { text: string; count: number; firstIndex: number }>();
  for (const [i, candidate] of candidates.entries()) {
    if (!candidate.text) continue;
    const key = normalizeForAgreement(candidate.text);
    const existing = counts.get(key);
    if (existing) existing.count++;
    else counts.set(key, { text: candidate.text, count: 1, firstIndex: i });
  }

  const ranked = [...counts.values()].sort((a, b) => b.count - a.count || a.firstIndex - b.firstIndex);
  const winner = ranked[0]?.text ?? "";
  const agreement = ranked[0]?.count ?? 0;
  const total = candidates.filter((candidate) => candidate.text).length;

  return { input, winner, agreement, total, candidates };
}

async function raceAutocorrect(input: string, models: string[], signal?: AbortSignal): Promise<RaceResult> {
  const candidates = await Promise.all(
    models.map(async (model): Promise<RaceCandidate> => {
      const start = Date.now();
      try {
        const text = await askOllama(input, model, signal);
        return { model, text, ms: Date.now() - start };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return { model, error: message, ms: Date.now() - start };
      }
    }),
  );

  const result = chooseRaceWinner(input, candidates);
  return { ...result, winner: result.winner || input };
}

async function racePrediction(input: string, models: string[], signal?: AbortSignal): Promise<RaceResult> {
  const candidates = await Promise.all(
    models.map(async (model): Promise<RaceCandidate> => {
      const start = Date.now();
      try {
        const text = await predictNextWithSpec(input, model, signal);
        return { model, text, ms: Date.now() - start };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return { model, error: message, ms: Date.now() - start };
      }
    }),
  );

  return chooseRaceWinner(input, candidates);
}

function storeRace(pi: ExtensionAPI, result: RaceResult, accepted?: boolean) {
  pi.appendEntry("ollama-autocorrect-race", {
    timestamp: Date.now(),
    accepted,
    input: result.input,
    winner: result.winner,
    agreement: result.agreement,
    total: result.total,
    candidates: result.candidates,
  });
}

function storePredictionRace(pi: ExtensionAPI, result: RaceResult, accepted?: boolean) {
  pi.appendEntry("ollama-prediction-race", {
    timestamp: Date.now(),
    accepted,
    input: result.input,
    winner: result.winner,
    agreement: result.agreement,
    total: result.total,
    candidates: result.candidates,
  });
}

function appendContinuation(text: string, continuation: string): string {
  if (!continuation) return text;
  if (!text || /\s$/.test(text) || /^\s|^[.,!?;:)]/.test(continuation)) return text + continuation;
  return `${text} ${continuation}`;
}

async function autocorrect(ctx: ExtensionContext, pi: ExtensionAPI, model = DEFAULT_MODEL) {
  if (!ctx.hasUI) return;

  const original = ctx.ui.getEditorText();
  if (!original.trim()) {
    ctx.ui.notify("Nothing to autocorrect.", "info");
    return;
  }

  ctx.ui.setStatus("ollama-autocorrect", ctx.ui.theme.fg("accent", "autocorrecting…"));

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);

  try {
    const corrected = await askOllama(original, model, controller.signal);
    if (!corrected) {
      ctx.ui.notify("Ollama returned no correction.", "warning");
      return;
    }

    ctx.ui.setEditorText(corrected);
    pi.appendEntry("ollama-autocorrect-single", {
      timestamp: Date.now(),
      model,
      input: original,
      output: corrected,
      accepted: true,
    });
    ctx.ui.notify(`Autocorrected with ${model}.`, "info");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    ctx.ui.notify(`Autocorrect failed: ${message}`, "error");
  } finally {
    clearTimeout(timeout);
    ctx.ui.setStatus("ollama-autocorrect", undefined);
  }
}

async function predict(ctx: ExtensionContext, pi: ExtensionAPI, model = PREDICTION_MODEL) {
  if (!ctx.hasUI) return;

  const original = ctx.ui.getEditorText();
  if (original.trim().length < PREDICTION_MIN_CHARS) {
    ctx.ui.notify("Type a little more before predicting.", "info");
    return;
  }

  ctx.ui.setStatus("ollama-autocorrect", ctx.ui.theme.fg("accent", "predicting next words…"));

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);

  try {
    const continuation = await predictNextWithSpec(original, model, controller.signal);
    if (!continuation) {
      ctx.ui.notify("Model returned no prediction.", "warning");
      return;
    }

    ctx.ui.setEditorText(appendContinuation(original, continuation));
    pi.appendEntry("ollama-autocorrect-predict", {
      timestamp: Date.now(),
      model,
      input: original,
      output: continuation,
      accepted: true,
    });
    ctx.ui.notify(`Predicted next text with ${model}.`, "info");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    ctx.ui.notify(`Prediction failed: ${message}`, "error");
  } finally {
    clearTimeout(timeout);
    ctx.ui.setStatus("ollama-autocorrect", undefined);
  }
}

async function predictionRace(ctx: ExtensionContext, pi: ExtensionAPI, models = PREDICTION_RACE_MODELS) {
  if (!ctx.hasUI) return;

  const original = ctx.ui.getEditorText();
  if (original.trim().length < PREDICTION_MIN_CHARS) {
    ctx.ui.notify("Type a little more before racing predictions.", "info");
    return;
  }

  ctx.ui.setStatus("ollama-autocorrect", ctx.ui.theme.fg("accent", `racing ${models.length} predictors…`));

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60_000);

  try {
    const result = await racePrediction(original, models, controller.signal);
    storePredictionRace(pi, result, Boolean(result.winner));

    if (!result.winner) {
      ctx.ui.notify("No predictor returned a continuation. Stored failures in session.", "warning");
      return;
    }

    ctx.ui.setEditorText(appendContinuation(original, result.winner));
    ctx.ui.notify(`Prediction race winner: ${result.agreement}/${result.total} agreement. Stored in session.`, "info");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    ctx.ui.notify(`Prediction race failed: ${message}`, "error");
  } finally {
    clearTimeout(timeout);
    ctx.ui.setStatus("ollama-autocorrect", undefined);
  }
}

async function autocorrectRace(ctx: ExtensionContext, pi: ExtensionAPI, models = FULL_RACE_MODELS) {
  if (!ctx.hasUI) return;

  const original = ctx.ui.getEditorText();
  if (!original.trim()) {
    ctx.ui.notify("Nothing to autocorrect.", "info");
    return;
  }

  ctx.ui.setStatus("ollama-autocorrect", ctx.ui.theme.fg("accent", `racing ${models.length} models…`));

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60_000);

  try {
    const result = await raceAutocorrect(original, models, controller.signal);
    ctx.ui.setEditorText(result.winner);
    storeRace(pi, result, true);
    ctx.ui.notify(`Race winner: ${result.agreement}/${result.total} agreement. Stored in session.`, "info");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    ctx.ui.notify(`Autocorrect race failed: ${message}`, "error");
  } finally {
    clearTimeout(timeout);
    ctx.ui.setStatus("ollama-autocorrect", undefined);
  }
}

export default function (pi: ExtensionAPI) {
  pi.registerCommand("autocorrect", {
    description: `Autocorrect the current editor text with Ollama (${DEFAULT_MODEL} by default). Usage: /autocorrect [model]`,
    handler: async (args, ctx) => {
      await autocorrect(ctx, pi, args.trim() || DEFAULT_MODEL);
    },
  });

  pi.registerCommand("autocorrect-race", {
    description: `Race Ollama autocorrect models and store agreement data. Usage: /autocorrect-race [model1,model2,...]`,
    handler: async (args, ctx) => {
      const models = args.trim()
        ? args.split(/[\s,]+/).map((m) => m.trim()).filter(Boolean)
        : FULL_RACE_MODELS;
      await autocorrectRace(ctx, pi, models);
    },
  });

  pi.registerCommand("predict", {
    description: `Predict the next text. Usage: /predict [ollama-model|ollama:model|openrouter:model]`,
    handler: async (args, ctx) => {
      await predict(ctx, pi, args.trim() || PREDICTION_MODEL);
    },
  });

  pi.registerCommand("predict-race", {
    description: `Race prediction providers. Usage: /predict-race [ollama:model,openrouter:model,...]`,
    handler: async (args, ctx) => {
      const models = args.trim()
        ? args.split(/[\s,]+/).map((m) => m.trim()).filter(Boolean)
        : PREDICTION_RACE_MODELS;
      await predictionRace(ctx, pi, models);
    },
  });

  pi.registerShortcut("ctrl+shift+a", {
    description: `Accept/autocorrect editor text with Ollama (${DEFAULT_MODEL})`,
    handler: async (ctx) => {
      await autocorrect(ctx, pi, DEFAULT_MODEL);
    },
  });

  pi.registerShortcut("ctrl+space", {
    description: `Predict and append next text with Ollama (${PREDICTION_MODEL})`,
    handler: async (ctx) => {
      await predict(ctx, pi, PREDICTION_MODEL);
    },
  });

  pi.on("session_start", (_event, ctx) => {
    if (!ctx.hasUI || ctx.mode !== "tui") return;

    class OllamaGhostEditor extends CustomEditor {
      private correctionGhost = "";
      private correctionMeta = "";
      private predictionGhost = "";
      private predictionMeta = "";
      private lastRace?: RaceResult;
      private lastRequested = "";
      private debounceTimer?: ReturnType<typeof setTimeout>;
      private abort?: AbortController;
      private requestId = 0;

      constructor(tui: TUI, theme: EditorTheme, keybindings: KeybindingsManager) {
        super(tui, theme, keybindings);
      }

      private clearGhost(): void {
        if (this.correctionGhost || this.correctionMeta || this.predictionGhost || this.predictionMeta) {
          this.correctionGhost = "";
          this.correctionMeta = "";
          this.predictionGhost = "";
          this.predictionMeta = "";
          this.lastRace = undefined;
          this.tui.requestRender();
        }
      }

      private scheduleGhost(): void {
        if (this.debounceTimer) clearTimeout(this.debounceTimer);
        this.abort?.abort();

        const text = this.getText();
        if (text.trim().length < GHOST_MIN_CHARS || text.trim().startsWith("/")) {
          this.clearGhost();
          return;
        }

        this.debounceTimer = setTimeout(() => {
          void this.refreshGhost(text);
        }, GHOST_DEBOUNCE_MS);
      }

      private async refreshGhost(text: string): Promise<void> {
        const trimmed = text.trim();
        if (!trimmed || trimmed === this.lastRequested) return;

        this.lastRequested = trimmed;
        const id = ++this.requestId;
        const controller = new AbortController();
        this.abort = controller;
        const predictionStart = Date.now();
        this.correctionGhost = "";
        this.correctionMeta = "";
        this.predictionGhost = "…";
        this.predictionMeta = ` ${PREDICTION_MODEL}`;
        this.tui.requestRender();

        try {
          const prediction = trimmed.length >= PREDICTION_MIN_CHARS
            ? await predictNext(text, PREDICTION_MODEL, controller.signal)
            : "";

          if (id !== this.requestId || controller.signal.aborted) return;

          if (prediction) {
            this.predictionGhost = prediction;
            this.predictionMeta = ` ${PREDICTION_MODEL} ${Date.now() - predictionStart}ms`;
          } else {
            this.predictionGhost = "";
            this.predictionMeta = "";
          }
        } catch (error) {
          if (id !== this.requestId) return;
          const message = error instanceof Error ? error.message : String(error);
          this.predictionGhost = "prediction unavailable";
          this.predictionMeta = ` ${PREDICTION_MODEL}: ${message.slice(0, 80)}`;
        }

        this.tui.requestRender();
      }

      handleInput(data: string): void {
        if (matchesKey(data, Key.tab) && this.correctionGhost && !this.isShowingAutocomplete()) {
          const accepted = this.lastRace;
          this.setText(this.correctionGhost);
          this.clearGhost();
          if (accepted) storeRace(pi, accepted, true);
          this.scheduleGhost();
          return;
        }

        if (matchesKey(data, Key.ctrl("space")) && this.predictionGhost && !this.isShowingAutocomplete()) {
          const input = this.getText();
          this.setText(appendContinuation(input, this.predictionGhost));
          pi.appendEntry("ollama-autocorrect-predict", {
            timestamp: Date.now(),
            model: PREDICTION_MODEL,
            input,
            output: this.predictionGhost,
            accepted: true,
          });
          this.clearGhost();
          this.scheduleGhost();
          return;
        }

        const before = this.getText();
        super.handleInput(data);
        const after = this.getText();
        if (after !== before) this.scheduleGhost();
      }

      render(width: number): string[] {
        const lines = super.render(width);
        if (lines.length < 2) return lines;

        const current = this.getText();
        const cursor = this.getCursor();
        const editorLines = this.getLines();
        const lastLineIndex = editorLines.length - 1;
        const cursorAtEnd = cursor.line === lastLineIndex && cursor.col === (editorLines[lastLineIndex]?.length ?? 0);
        let renderedLines = lines;

        if (this.predictionGhost && cursorAtEnd) {
          const cursorLineIndex = lines.findIndex((line) => line.includes(CURSOR_MARKER));
          if (cursorLineIndex >= 0) {
            const cursorSpace = "\x1b[7m \x1b[0m";
            const line = lines[cursorLineIndex]!;
            const markerIndex = line.indexOf(CURSOR_MARKER);
            const beforeMarker = line.slice(0, markerIndex);
            const afterMarker = line.slice(markerIndex + CURSOR_MARKER.length);
            const afterCursor = afterMarker.startsWith(cursorSpace) ? afterMarker.slice(cursorSpace.length) : afterMarker;
            const inlineGhost = ctx.ui.theme.fg("dim", this.predictionGhost);
            renderedLines = [...lines];
            renderedLines[cursorLineIndex] = `${beforeMarker}${CURSOR_MARKER}${inlineGhost}${afterCursor}`;
          }
        }

        const ghostLines: string[] = [];

        if (this.correctionGhost && this.correctionGhost !== current) {
          ghostLines.push(
            ...wrapTextWithAnsi(ctx.ui.theme.fg("dim", `↳ fix: ${this.correctionGhost}${this.correctionMeta}`), width),
          );
        }

        if (this.predictionGhost && !cursorAtEnd) {
          ghostLines.push(...wrapTextWithAnsi(ctx.ui.theme.fg("dim", `↳ next: ${this.predictionGhost}`), width));
        }

        if (!ghostLines.length) return renderedLines;

        return [...renderedLines.slice(0, -1), ...ghostLines, renderedLines[renderedLines.length - 1]!];
      }
    }

    ctx.ui.setEditorComponent((tui, theme, keybindings) => new OllamaGhostEditor(tui, theme, keybindings));
    ctx.ui.notify("Ollama prediction loaded: one live model, Ctrl+Space accepts inline ghost", "info");
  });
}
