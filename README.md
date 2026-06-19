# pi-ollama-autocorrect

Local Ollama-powered autocorrect for the Pi textbox.

## Features

- dim gray ghost correction while you type
- second ghost line predicts what you may type next
- `Tab` accepts the autocorrect ghost
- `Ctrl+Space` accepts the prediction ghost
- `Ctrl+Shift+A` runs/apply autocorrect manually
- `/autocorrect [model]`
- `/predict [model]`
- `/autocorrect-race [model1,model2,...]` stores model agreement data in the Pi session

## Requirements

- Pi coding agent
- Ollama running locally on `127.0.0.1:11434`
- default autocorrect/live fix model: `llama3.2:latest`
- default prediction model: `qwen2.5-coder:1.5b`

Start Ollama:

```bash
ollama serve
```

Pull the default model if needed:

```bash
ollama pull llama3.2
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
