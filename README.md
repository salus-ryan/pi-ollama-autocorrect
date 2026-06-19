# pi-ollama-autocorrect

Local Ollama-powered autocorrect/prediction for the Pi textbox, with optional OpenRouter prediction races.

## Features

- one live Ollama model while typing for better responsiveness
- dim gray inline prediction at the cursor
- `Ctrl+Space` accepts the prediction ghost
- `Tab` accepts an autocorrect ghost when available
- `Ctrl+Shift+A` runs/apply autocorrect manually
- `/autocorrect [model]`
- `/predict [model]`
- `/predict openrouter:openai/gpt-4o-mini` uses OpenRouter for one prediction
- `/predict-race [ollama:model,openrouter:model,openrouter:auto,...]` races local/API predictors and stores results
- `/openrouter-models [limit]` fetches OpenRouter pricing and stores a cost/value ranking
- local AI-native typing telemetry: shown, accepted, typed-past, cleared, latency, cost estimates
- `/typing-telemetry-stats`, `/typing-telemetry-export`, `/typing-telemetry-clear`
- `/autocorrect-race [model1,model2,...]` stores model agreement data in the Pi session

## Requirements

- Pi coding agent
- Ollama running locally on `127.0.0.1:11434`
- default autocorrect/live fix model: `llama3.2:latest`
- default live prediction model: `qwen2.5:0.5b`
- optional OpenRouter API key in `OPENROUTER_API_KEY`

Start Ollama:

```bash
ollama serve
```

Pull the default models if needed:

```bash
ollama pull llama3.2
ollama pull qwen2.5:0.5b
```

For OpenRouter prediction races/pricing:

```bash
export OPENROUTER_API_KEY=your_key_here
```

Cost-aware race example:

```text
/predict-race ollama:qwen2.5:0.5b,ollama:llama3.2:latest,openrouter:auto
```

## Install locally

From this repo:

```bash
pi install git:/absolute/path/to/pi-ollama-autocorrect
```

Or copy `extensions/index.ts` to:

```text
~/.pi/agent/extensions/ollama-autocorrect.ts
```

Then reload Pi:

```text
/reload
```

## Telemetry / training data

The extension appends structured local session events for future evaluation/fine-tuning:

- suggestion requested/shown
- accepted via `Tab` or `Ctrl+Space`
- typed-past / cleared suggestions
- latency, model/provider, rough token estimates
- prediction/autocorrect race results

Export current in-memory events to a JSONL session entry:

```text
/typing-telemetry-export
```

Telemetry is redacted by default for common API keys/tokens. To keep raw text for a controlled local experiment:

```bash
export PI_AUTOCORRECT_RAW_TELEMETRY=1
```

## Notes

This is intentionally an extension rather than a Pi core feature. Autocorrect depends on local model availability, latency, and personal preference.
