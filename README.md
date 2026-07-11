# vibeVAServer

Vibe implementation of the `AudioCloneModelWorker` gRPC service from
`audio-interface/proto/clone-interface.proto`.

## Run

```bash
# Foreground server
uv run vibevaserver-start --port 50051

# Background daemon
uv run vibevaserver-start --daemon --port 50051

# Stop daemon
uv run vibevaserver-stop
```

These commands are exposed by `[project.scripts]` in `pyproject.toml`:

| Command | Entrypoint |
|---------|------------|
| `vibevaserver-start` | `start_server:main` |
| `vibevaserver-stop` | `stop_vibe:main` |

## Architectural Decision Log

### Spawn-based daemon entrypoint and lazy app import

Context: [Daemon Import Caveat](https://github.com/NeechLog/audio-interface/blob/main/docs/daemon-import-caveat.md)

`vibevaserver-start` uses the shared `audiocloneserver`
`run_service_entrypoint(...)` helper. The start script passes `app:main` as an
import string so daemon handling can run before importing `app.py`.

Daemon mode starts a detached fresh child process with daemon mode disabled in
the child. The child then imports the Vibe app and starts the foreground gRPC
server. This avoids fork-based daemonization after model, gRPC, or other native
runtime state may have been initialized.
