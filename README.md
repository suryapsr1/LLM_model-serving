# Async LLM Serving Gateway

FastAPI streams responses from a local GGUF model, and Celery processes queued inference jobs through Redis. An optional OpenAI-compatible cloud engine can handle requests when configured. The API documentation is at <http://127.0.0.1:8000/docs>.

## Requirements

- Docker Desktop running Linux containers, with Docker Compose available.
- The GGUF model at [`models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`](./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf), or a different directory and filename configured in [`.env.example`](./.env.example). The supplied model is 807,694,464 bytes (about 808 MB).
- Enough Docker memory for **two model instances** plus their context buffers and service overhead: one in the API and another in the worker. The worker loads its model on its first local job. Start with 4 GB available to Docker and adjust for your context size and workload.
- Network access for the first image build. It prefers a CPU wheel for the pinned llama-cpp-python release; if no compatible wheel is available, the builder compiles it. That first compilation can take several minutes. Build tools are omitted from the final runtime image.

## Start the full project

From this project directory in PowerShell:

```powershell
# Optional: customize the settings before starting.
if (-not (Test-Path .env)) { Copy-Item .env.example .env }

.\scripts\compose.ps1 up --build --detach --wait --wait-timeout 180
.\scripts\compose.ps1 ps
```

The PowerShell helper locates Docker Desktop and adds its CLI directory to the process PATH, including the credential helper needed to pull images. This handles per-user Docker installations that are missing from PATH. On systems where Docker is already configured, the equivalent command is:

```sh
docker compose up --build --detach --wait --wait-timeout 180
```

The API listens on **127.0.0.1:8000**. Redis is available only inside the Compose network. Model files are mounted read-only and excluded from the image. Redis jobs and results use a persistent named volume with append-only persistence; default Redis fsync can still lose the last second of writes after an abrupt host failure.

The API health check returns an error if neither inference engine is available. Worker health checks verify Celery responds through Redis; a successful batch smoke test below also verifies that its model can load and generate text. If a configured cloud key is invalid, the provider error is reported during inference rather than during startup.

For cloud fallback, set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` in your local `.env`, then recreate the services with the start command. Cloud requests may incur provider charges. Keep `.env` private. No cloud key is needed for local inference.

## Deploy on Railway

Connect this GitHub repository to an API service and build it with the included
Dockerfile. The container listens on `0.0.0.0` and uses `PORT`, defaulting to 8000.
To keep port 8000 on Railway, set the service variable `PORT=8000` and set the
public domain's target port to 8000 under Settings > Networking. Generate a
public domain if the service does not have one. Open `https://<your-domain>/docs`
for the API interface; the root `/` has no page. Public HTTPS uses port 443, so
do not append `:8000` to the public URL.

Configure an inference engine before enabling the `/healthz` deployment health
check:

- For the local engine, provision the GGUF model on a Railway volume and set
  `LOCAL_MODEL_PATH` to its absolute path. The model is excluded from Git and the
  Docker image; your computer's `models/` directory is not uploaded to Railway.
  Ensure the container's user (UID 10001) can read the model.
- Alternatively, set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` in
  Railway's service variables for the OpenAI-compatible cloud engine. Cloud
  inference may incur provider charges. Select `engine: "cloud"` for batch jobs.

Without either engine, `/healthz` returns 503 and inference is unavailable.
Use `/healthz` as the health check path once an engine is configured.

Batch jobs also need a Redis service and a separate worker service using the
same repository/image. Set `REDIS_URL` to the Redis connection URL on both the
API and worker. Set the worker's start command to
`celery -A app.worker.celery_app:celery_app worker --loglevel=info --concurrency=1`.
Configure the inference engine on both services. A local worker also needs
access to its model file; do not assume it shares the API's volume. Leave the
worker's HTTP health check unset. Streaming does not require a worker or Redis.

Redeploy after updating source or service variables, then verify `/healthz`,
`/docs`, and an inference request. A GitHub push alone does not create or connect
a Railway service.

## Smoke checks

These requests use the local engine. Change the port if you set `API_PORT`.

```powershell
$baseUrl = 'http://127.0.0.1:8000'
Invoke-RestMethod "$baseUrl/healthz"

$body = @{
    prompt = 'Reply with a short greeting.'
    max_tokens = 24
    temperature = 0
    top_p = 0.9
    engine = 'local'
} | ConvertTo-Json

# A successful SSE response contains text chunks and a final is_eos event.
$stream = Invoke-WebRequest "$baseUrl/v1/chat/stream" -Method Post -ContentType 'application/json' -Body $body -UseBasicParsing
$stream.Content

# Queue a job, then poll for SUCCESS or FAILURE for up to two minutes.
$job = Invoke-RestMethod "$baseUrl/v1/batch/submit" -Method Post -ContentType 'application/json' -Body $body
$deadline = (Get-Date).AddMinutes(2)
do {
    Start-Sleep -Seconds 2
    $result = Invoke-RestMethod "$baseUrl/v1/batch/status/$($job.task_id)"
} while ($result.state -notin @('SUCCESS', 'FAILURE', 'REVOKED') -and (Get-Date) -lt $deadline)
$result | ConvertTo-Json -Depth 5
if ($result.state -ne 'SUCCESS') { throw "Batch did not succeed: $($result.state)" }
```

`Invoke-WebRequest` buffers the displayed response. The API itself streams SSE; clients can consume each `data:` event as it arrives. For automated checks with the existing Python environment:

```powershell
# Regression tests use fake engines and HTTP transports; no API keys or Redis needed.
.\venv\Scripts\python.exe -m unittest discover -s tests -v
# Load the real model directly.
.\venv\Scripts\python.exe test_local.py
# Check health, real streaming, and a real Redis/Celery batch job against the running stack.
.\venv\Scripts\python.exe test_api.py
```

The smoke scripts exit with an error if generation fails, produces empty output, or lacks a completion event. The API smoke script requires a healthy local engine. Batch results expire after one hour; unknown or expired task IDs report `PENDING` because Celery cannot distinguish them from queued work.

## Logs and shutdown

```powershell
.\scripts\compose.ps1 logs --tail 100 api worker redis
.\scripts\compose.ps1 down
```

Shutdown preserves Redis data. Avoid `down --volumes` unless you intend to delete queued jobs and stored results. After changing application code, use the start command with `--build` again. After changing only `.env`, `up --detach --wait` recreates services with the new configuration.

## Load test with Locust

Start the full Compose stack and check that `http://127.0.0.1:8000/healthz` returns HTTP 200 before starting the load test. Install the separate benchmark dependency into the virtual environment if needed:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt
.\venv\Scripts\python.exe -m locust -f benchmark_locust.py --host http://127.0.0.1:8000 --web-host 127.0.0.1
```

Open the Locust dashboard at <http://127.0.0.1:8089>. Port 8000 is the API being tested; port 8089 is Locust's web interface. Start with one user and a spawn rate of one user per second before increasing load. The benchmark checks health, consumes the full SSE stream, and enqueues batch jobs. Batch submission only measures queue acceptance, not job completion. Stop the test before shutting down the stack.

For a short terminal-only verification:

```powershell
.\venv\Scripts\python.exe -m locust -f benchmark_locust.py --host http://127.0.0.1:8000 --headless --users 1 --spawn-rate 1 --run-time 30s --exit-code-on-error 1 --only-summary
```

If startup fails, check Docker Desktop is using Linux containers, the model file exists and is readable, and port 8000 is free. The API logs show model-loading errors. Increase `LOCAL_N_CTX` only when needed because it increases memory use; `LOCAL_N_THREADS` controls CPU threads per model instance.

## Run the API without Docker

Python 3.11 is the Docker runtime and the tested local interpreter. With dependencies from [`requirements.txt`](./requirements.txt) installed in a virtual environment, start the API from the project root:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Local streaming works without Redis. Batch endpoints additionally require a reachable Redis instance and a Celery worker configured with the same `REDIS_URL`. Docker Compose is the supported way to run the complete stack on Windows; Celery's default prefork worker is intended for Linux. Local commands read process environment variables; Compose's `.env` file is not loaded automatically by the Python application.

## Resolved flaws

- The image now starts the API by default, and the native toolchain is confined to a separate builder stage. Direct dependency versions are pinned in [`requirements.txt`](./requirements.txt).
- The clean container exposed missing Redis transport dependencies, including `packaging`. Declaring `celery[redis]` now installs the full transport dependencies instead of relying on unrelated packages in a developer environment.
- Compose passes cloud and model settings to both app services, reports readiness, keeps Redis internal, and persists queued jobs and results.
- The model and local environment files are excluded from the build context, and the runtime runs as an unprivileged user.
- Concurrent local streams previously entered the same mutable llama.cpp context. Generation is now serialized per model, including native work finishing after a client disconnects. The API remains responsive while inference runs in a separate thread.
- Worker exceptions and terminal error chunks previously became successful tasks with empty or partial output. They now produce Celery `FAILURE` and a JSON-safe error in the status endpoint. Batch requests retain `top_p` and return HTTP 202.
- Synchronous Redis calls previously ran on the API event loop. Batch handlers now run in FastAPI's thread pool, with bounded Redis timeouts and HTTP 503 on queue failures.
- Invalid engine names and explicit null sampling values are rejected with HTTP 422. Health returns HTTP 503 when no engine can be initialized.
- Cloud streams now handle standard SSE variations and reject provider errors or truncated output. Streaming falls back to cloud only when local generation has not emitted text; batch jobs use the explicitly requested engine. Cloud behavior is verified with mocked responses without provider charges.
- Local model paths resolve from the project location, model resources close at shutdown, and the direct model smoke script uses the correct import and a practical token limit.

The service has no API authentication and is bound to the local machine by default. Add an authenticated gateway before making it reachable by other users.
