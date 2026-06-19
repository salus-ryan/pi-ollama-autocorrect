import {
  CustomEditor,
  type ExtensionAPI,
  type ExtensionContext,
  type KeybindingsManager,
} from "@earendil-works/pi-coding-agent";
import type { EditorTheme, TUI } from "@earendil-works/pi-tui";
import { Key, matchesKey, wrapTextWithAnsi } from "@earendil-works/pi-tui";

const DEFAULT_MODEL = "llama3.2:latest";
const OLLAMA_URL = "http://127.0.0.1:11434/api/generate";
const GHOST_DEBOUNCE_MS = 900;
const GHOST_MIN_CHARS = 8;
const PREDICTION_MIN_CHARS = 16;

// Fast enough for live ghost text. /autocorrect-race uses the longer list below.
// Live racing multiple models made the textbox feel broken on-device because
// Ollama queues/contends the generations. Keep live ghost fast; race manually.
const GHOST_MODEL = "llama3.2:latest";
const PREDICTION_MODEL = "qwen2.5-coder:1.5b";
const FULL_RACE_MODELS = ["qwen2.5-coder:1.5b", "llama3.2:latest", "mistral:latest", "phi:latest"];

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
  const raw = data.response?.trim() ?? "";
  const withoutQuotes = raw.replace(/^['"“”]+|['"“”]+$/g, "");
  return withoutQuotes.startsWith(original) ? withoutQuotes.slice(original.length).trim() : withoutQuotes;
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

  const counts = new Map<string, { text: string; count: number; firstIndex: number }>();
  for (const [i, candidate] of candidates.entries()) {
    if (!candidate.text) continue;
    const key = normalizeForAgreement(candidate.text);
    const existing = counts.get(key);
    if (existing) existing.count++;
    else counts.set(key, { text: candidate.text, count: 1, firstIndex: i });
  }

  const ranked = [...counts.values()].sort((a, b) => b.count - a.count || a.firstIndex - b.firstIndex);
  const winner = ranked[0]?.text ?? input;
  const agreement = ranked[0]?.count ?? 0;
  const total = candidates.filter((candidate) => candidate.text).length;

  return { input, winner, agreement, total, candidates };
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
    const continuation = await predictNext(original, model, controller.signal);
    if (!continuation) {
      ctx.ui.notify("Ollama returned no prediction.", "warning");
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
    description: `Predict the next text with Ollama (${PREDICTION_MODEL} by default). Usage: /predict [model]`,
    handler: async (args, ctx) => {
      await predict(ctx, pi, args.trim() || PREDICTION_MODEL);
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
        const correctionStart = Date.now();
        const predictionStart = Date.now();
        this.predictionGhost = "…";
        this.predictionMeta = ` ${PREDICTION_MODEL}`;
        this.tui.requestRender();

        const [correction, prediction] = await Promise.allSettled([
          askOllama(text, GHOST_MODEL, controller.signal),
          trimmed.length >= PREDICTION_MIN_CHARS
            ? predictNext(text, PREDICTION_MODEL, controller.signal)
            : Promise.resolve(""),
        ]);

        if (id !== this.requestId || controller.signal.aborted) return;

        if (correction.status === "fulfilled") {
          const corrected = correction.value;
          const candidate: RaceCandidate = { model: GHOST_MODEL, text: corrected, ms: Date.now() - correctionStart };
          const result: RaceResult = {
            input: text,
            winner: corrected || text,
            agreement: corrected ? 1 : 0,
            total: corrected ? 1 : 0,
            candidates: [candidate],
          };
          this.lastRace = result;
          storeRace(pi, result, false);

          if (corrected && corrected !== text) {
            this.correctionGhost = corrected;
            this.correctionMeta = ` ${GHOST_MODEL} ${candidate.ms}ms`;
          } else {
            this.correctionGhost = "";
            this.correctionMeta = "";
          }
        } else {
          this.correctionGhost = "";
          this.correctionMeta = "";
        }

        if (prediction.status === "fulfilled" && prediction.value) {
          this.predictionGhost = prediction.value;
          this.predictionMeta = ` ${PREDICTION_MODEL} ${Date.now() - predictionStart}ms`;
        } else if (prediction.status === "rejected") {
          const message = prediction.reason instanceof Error ? prediction.reason.message : String(prediction.reason);
          this.predictionGhost = "prediction unavailable";
          this.predictionMeta = ` ${PREDICTION_MODEL}: ${message.slice(0, 80)}`;
        } else {
          this.predictionGhost = "";
          this.predictionMeta = "";
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
        const ghostLines: string[] = [];

        if (this.correctionGhost && this.correctionGhost !== current) {
          ghostLines.push(
            ...wrapTextWithAnsi(ctx.ui.theme.fg("dim", `↳ fix: ${this.correctionGhost}${this.correctionMeta}`), width),
          );
        }

        if (this.predictionGhost) {
          ghostLines.push(
            ...wrapTextWithAnsi(ctx.ui.theme.fg("dim", `↳ next: ${this.predictionGhost}${this.predictionMeta}`), width),
          );
        }

        if (!ghostLines.length) return lines;

        // True inline gray text behind the cursor is not exposed by pi's editor API yet,
        // so render the live Ollama correction/prediction as dim wrapped lines above the bottom border.
        return [...lines.slice(0, -1), ...ghostLines, lines[lines.length - 1]!];
      }
    }

    ctx.ui.setEditorComponent((tui, theme, keybindings) => new OllamaGhostEditor(tui, theme, keybindings));
    ctx.ui.notify("Ollama autocorrect loaded: Tab accepts fix, Ctrl+Space accepts prediction", "info");
  });
}
