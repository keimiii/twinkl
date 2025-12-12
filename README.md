# Twinkl

Academic capstone project (NUS MTech IS). This repo uses `uv` and `pyproject.toml`
for dependency management.

## Setup

1. Install `uv` (see https://docs.astral.sh/uv/).
2. Create the virtual environment:
   ```sh
   uv venv
   ```
3. Activate it (Fish shell):
   ```sh
   source .venv/bin/activate.fish
   ```

## Installing dependencies

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

- Install everything from the lockfile:
  ```sh
  uv sync
  ```
- If/when a dev group is added later:
  ```sh
  uv sync --dev
  ```

## Adding a dependency

Use `uv add` to both install into the environment and record it in
`pyproject.toml`:

```sh
uv add <package>
```

Pin an exact version if desired:

```sh
uv add "<package>==<version>"
```

After adding, `uv` updates `uv.lock` automatically.

## Exporting requirements.txt (optional)

Only needed for legacy tooling or platforms that require it:

```sh
uv export -o requirements.txt
```
