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
      const url = String(input);
      const path = url.replace("http://localhost:8000", "");
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

test("navigation renders tabs and routes", async () => {
  mockFetch({
    "/api/dashboard": {
      system_state: {
        offline_mode: true,
        mlflow_logging: false,
        strict_determinism: true,
        run_controls_enabled: false,
      },
      last_run: null,
      aggregate_metrics: {},
      meetings: [],
    },
  });
  renderApp("/");
  expect(screen.getByText("Dashboard")).toBeInTheDocument();
  expect(screen.getByText("Meetings")).toBeInTheDocument();
  expect(screen.getByText("Evaluation")).toBeInTheDocument();
  await waitFor(() => expect(screen.getByText("System Health")).toBeInTheDocument());
});
