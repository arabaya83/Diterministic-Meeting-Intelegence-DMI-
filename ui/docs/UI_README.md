# AMI UI README

## Architecture

```text
Browser (React + Vite dev server or built static assets)
        |
        |  localhost HTTP
        v
FastAPI backend (ui/backend)
        |
        +-- reads repository-local artifacts and configs
            artifacts/
            data/staged/
            configs/
            docs/
```

Development:
- Frontend dev server on `http://localhost:5173`
- Backend API on `http://localhost:8000`

Production-like local deployment:
- `make ui-build` builds `ui/frontend/dist`
- FastAPI serves the built frontend from `ui/frontend/dist`

## How To Run Dev

Backend only:

```bash
make ui-backend
```

Frontend only:

```bash
make ui-frontend
```

Both together:

```bash
make ui-dev
```

## How To Run Tests

Backend tests:

```bash
PYTHONPATH=ui/backend python3 -m pytest -q ui/backend/tests
```

Frontend tests:

```bash
npm --prefix ui/frontend install
npm --prefix ui/frontend run test
```

Combined:

```bash
make ui-test
```

Smoke validation:

```bash
bash ui/scripts/smoke_test.sh
```

## Backend Path Configuration

The backend defaults are repository-local and offline-first. Override them with environment variables when needed:

- `AMI_UI_PROJECT_ROOT`
- `AMI_UI_RAW_AMI_AUDIO_DIR`
- `AMI_UI_ARTIFACTS_DIR`
- `AMI_UI_EVAL_DIR`
- `AMI_UI_CONFIGS_DIR`
- `AMI_UI_DOCS_DIR`
- `AMI_UI_STAGED_DIR`
- `AMI_UI_FRONTEND_DIST_DIR`
- `AMI_UI_ENABLE_RUN_CONTROLS`

Example:

```bash
AMI_UI_ARTIFACTS_DIR=/tmp/ami-artifacts make ui-backend
```

## Known Limitations

- Waveform rendering is not included in V1; the Speech tab uses HTML5 audio plus segment tables.
- Run and validate controls are scaffolded but disabled by default.
- The configuration page is read-only in V1.
