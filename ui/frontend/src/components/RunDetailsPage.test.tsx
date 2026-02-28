import { screen, waitFor } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { renderApp } from "../test/render";

afterEach(() => {
  vi.restoreAllMocks();
});

function mockFetch(jsonByPath: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const path = String(input).replace("http://localhost:8000", "");
      if (!(path in jsonByPath)) {
        return Promise.resolve(new Response("Not Found", { status: 404 }));
      }
      return Promise.resolve(
        new Response(JSON.stringify(jsonByPath[path]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    }),
  );
}

test("run details page renders progress and logs", async () => {
  mockFetch({
    "/api/runs/run123": {
      run_id: "run123",
      meeting_id: "ES2005a",
      config: "pipeline.nemo.llama.final_eval.yaml",
      mode: "run",
      status: "running",
      started_at: "2026-02-28T12:00:00Z",
      ended_at: null,
      command: ["python3", "scripts/run_nemo_batch_sequential.py"],
      recent_logs: ["starting", "working"],
      exit_code: null,
      summary: null,
      stage_events: [{ event: "stage_start", stage: "asr" }],
      artifact_digest: null,
      progress: {
        current_stage_key: "asr",
        current_stage_name: "ASR",
        completed_stages: 3,
        total_stages: 11,
        last_event: "Started ASR",
        stages: [
          { key: "ingest", name: "Ingest", status: "completed", runtime_sec: 0.1, summary: null },
          { key: "vad", name: "VAD", status: "completed", runtime_sec: 0.2, summary: null },
          { key: "diarization", name: "Diarization", status: "completed", runtime_sec: 1.2, summary: null },
          { key: "asr", name: "ASR", status: "running", runtime_sec: null, summary: null },
        ],
      },
    },
  });

  renderApp("/runs/run123");
  await waitFor(() => expect(screen.getByText("Run run123")).toBeInTheDocument());
  expect(screen.getAllByText("ASR").length).toBeGreaterThan(0);
  expect(screen.getByText("Recent Logs")).toBeInTheDocument();
  expect(screen.getAllByText((_, element) => element?.textContent?.includes("starting") ?? false).length).toBeGreaterThan(0);
});
