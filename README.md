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
- `/predict-race [ollama:model,openrouter:model,...]` races local/API predictors and stores results
- `/autocorrect-race [model1,model2,...]` stores model agreement data in the Pi session

## Requirements

- Pi coding agent
- Ollama running locally on `127.0.0.1:11434`
- default autocorrect/live fix model: `llama3.2:latest`
- default live prediction model: `llama3.2:latest`
- optional OpenRouter API key in `OPENROUTER_API_KEY`

Start Ollama:

```bash
ollama serve
```

Pull the default model if needed:

```bash
ollama pull llama3.2
```

For OpenRouter prediction races:

```bash
export OPENROUTER_API_KEY=your_key_here
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

## Notes

This is intentionally an extension rather than a Pi core feature. Autocorrect depends on local model availability, latency, and personal preference.
