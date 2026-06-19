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

// Fast enough for live ghost text. /autocorrect-race uses the longer list below.
// Live racing multiple models made the textbox feel broken on-device because
// Ollama queues/contends the generations. Keep live ghost fast; race manually.
const GHOST_MODEL = "llama3.2:latest";
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

  pi.registerShortcut("ctrl+shift+a", {
    description: `Accept/autocorrect editor text with Ollama (${DEFAULT_MODEL})`,
    handler: async (ctx) => {
      await autocorrect(ctx, pi, DEFAULT_MODEL);
    },
  });

  pi.on("session_start", (_event, ctx) => {
    if (!ctx.hasUI || ctx.mode !== "tui") return;

    class OllamaGhostEditor extends CustomEditor {
      private ghost = "";
      private ghostMeta = "";
      private lastRace?: RaceResult;
      private lastRequested = "";
      private debounceTimer?: ReturnType<typeof setTimeout>;
      private abort?: AbortController;
      private requestId = 0;

      constructor(tui: TUI, theme: EditorTheme, keybindings: KeybindingsManager) {
        super(tui, theme, keybindings);
      }

      private clearGhost(): void {
        if (this.ghost || this.ghostMeta) {
          this.ghost = "";
          this.ghostMeta = "";
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
        const start = Date.now();

        try {
          const corrected = await askOllama(text, GHOST_MODEL, controller.signal);
          if (id !== this.requestId || controller.signal.aborted) return;

          const candidate: RaceCandidate = { model: GHOST_MODEL, text: corrected, ms: Date.now() - start };
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
            this.ghost = corrected;
            this.ghostMeta = ` ${GHOST_MODEL} ${candidate.ms}ms`;
          } else {
            this.ghost = "";
            this.ghostMeta = "";
          }
          this.tui.requestRender();
        } catch {
          if (id === this.requestId) this.clearGhost();
        }
      }

      handleInput(data: string): void {
        if (matchesKey(data, Key.tab) && this.ghost && !this.isShowingAutocomplete()) {
          const accepted = this.lastRace;
          this.setText(this.ghost);
          this.clearGhost();
          if (accepted) storeRace(pi, accepted, true);
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
        if (!this.ghost || lines.length < 2) return lines;

        const current = this.getText();
        if (this.ghost === current) return lines;

        const ghostText = `↳ ${this.ghost}${this.ghostMeta}`;
        const ghostLines = wrapTextWithAnsi(ctx.ui.theme.fg("dim", ghostText), width);

        // True inline gray text behind the cursor is not exposed by pi's editor API yet,
        // so render the live Ollama correction as dim wrapped lines above the bottom border.
        return [...lines.slice(0, -1), ...ghostLines, lines[lines.length - 1]!];
      }
    }

    ctx.ui.setEditorComponent((tui, theme, keybindings) => new OllamaGhostEditor(tui, theme, keybindings));
    ctx.ui.notify("Ollama autocorrect loaded: fast ghost + /autocorrect-race for horse race", "info");
  });
}
