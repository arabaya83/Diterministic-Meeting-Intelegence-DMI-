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

test("run history page renders live and historical runs", async () => {
  mockFetch({
    "/api/runs": [
      {
        run_id: "run123",
        meeting_id: "ES2005a",
        config: "pipeline.nemo.llama.final_eval.yaml",
        mode: "run",
        status: "running",
        started_at: "2026-02-28T12:00:00Z",
        ended_at: null,
        artifact_digest: null,
        source: "live",
      },
      {
        run_id: null,
        meeting_id: "IS1001d",
        config: "configs/pipeline.nemo.yaml",
        mode: "validate-only",
        status: "not_run",
        started_at: "2026-02-28T11:00:00Z",
        ended_at: "2026-02-28T11:00:00Z",
        artifact_digest: null,
        source: "history",
      },
    ],
  });

  renderApp("/runs");
  await waitFor(() => expect(screen.getByText("Run History")).toBeInTheDocument());
  expect(screen.getByText("ES2005a")).toBeInTheDocument();
  expect(screen.getByText("IS1001d")).toBeInTheDocument();
});
