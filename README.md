# Sensai

## Requirements

- [uv](https://docs.astral.sh/uv/)
- `make`

uv downloads the right Python version (3.14, see `.python-version`) if it is
not already installed.

## Quick start

```sh
make          # create .venv, install dependencies and the pre-commit hooks
make run      # run the program
```

You can also call uv directly, without activating the virtualenv:

```sh
uv run sensai             # run the program
```

## Development

```sh
make tests    # run the test suite
make lint     # run every pre-commit check (same as the CI)
make format   # format the code and fix what ruff can fix
```

The coding style (80-character lines, every ruff rule enabled) is configured
in `pyproject.toml` under `[tool.ruff]`. The pre-commit hooks, defined in
`.pre-commit-config.yaml`, run on every commit:

- ruff check and ruff format
- trailing whitespace, end of file, YAML/TOML syntax, merge conflicts, large
  files
- [Conventional Commits](https://www.conventionalcommits.org/) format for
  commit messages (`feat: ...`, `fix: ...`)

The CI rejects any code that does not pass these checks or the tests.

## Make targets

| Target    | Description                                                     |
|-----------|-----------------------------------------------------------------|
| `all`     | Default. Runs `install`                                         |
| `install` | `uv sync --locked`, then installs the pre-commit hooks          |
| `run`     | Runs the program                                                |
| `tests`   | Runs the test suite with `pytest`                               |
| `lint`    | Runs every pre-commit check on all files                        |
| `format`  | Runs `ruff format` and `ruff check --fix`                       |
| `lock`    | Regenerates `uv.lock` from `pyproject.toml`                     |

## Dependencies

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`:

- `[project].dependencies`: runtime dependencies, needed to run the
  program. These are the only ones included when the package is built.
- `[dependency-groups].dev`: development tools (pytest, ruff, pre-commit).
- `uv.lock`: every package, including transitive ones, at an exact version.
  It is generated, do not edit it, but commit it.

`uv sync` installs both runtime and dev dependencies. Use
`uv sync --no-dev` to install only what the program needs.

To add or upgrade a dependency:

```sh
uv add <package>                  # runtime dependency
uv add --dev <package>            # dev dependency
uv lock --upgrade-package <pkg>   # upgrade one package
```

Then commit `pyproject.toml` and `uv.lock`. `make install` (and the CI) use
`--locked`: they fail if `uv.lock` is out of date instead of resolving new
versions.

## Project layout

```
.
├── pyproject.toml            # project metadata, dependencies, tool config
├── uv.lock                   # locked dependencies (generated)
├── .python-version           # Python version used by uv
├── .pre-commit-config.yaml   # pre-commit hooks
├── Makefile
├── .github/workflows/        # CI: install, run, style, tests + mirroring
├── src/
│   └── sensai/               # the package
│       ├── __init__.py
│       ├── __main__.py       # `python -m sensai`
│       └── main.py           # entry point: main()
├── tests/                    # pytest test suite
└── docs/
```
