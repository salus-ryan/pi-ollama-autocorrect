# pi-ollama-autocorrect

Local Ollama-powered autocorrect for the Pi textbox.

## Features

- dim gray ghost correction while you type
- `Tab` accepts the ghost suggestion
- `Ctrl+Shift+A` runs/apply autocorrect manually
- `/autocorrect [model]`
- `/autocorrect-race [model1,model2,...]` stores model agreement data in the Pi session

## Requirements

- Pi coding agent
- Ollama running locally on `127.0.0.1:11434`
- default live model: `llama3.2:latest`

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
