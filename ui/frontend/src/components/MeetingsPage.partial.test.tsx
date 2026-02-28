import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { renderApp } from "../test/render";

afterEach(() => {
  vi.restoreAllMocks();
});

function installFetchMock(map: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = String(input).replace("http://localhost:8000", "");
      const payload = map[url];
      if (payload === undefined) {
        return Promise.resolve(new Response("{}", { status: 404 }));
      }
      return Promise.resolve(
        new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    }),
  );
}

test("partial meetings show missing transcript guidance instead of broken preview requests", async () => {
  installFetchMock({
    "/api/meetings": [
      {
        meeting_id: "IS1001d",
        has_raw_audio: true,
        has_staged_audio: false,
        has_artifacts: false,
        last_updated: "2026-02-28T12:00:00Z",
        config_digest: null,
        artifact_digest: null,
        offline_preflight_ok: null,
        determinism_risks: [],
        stages_complete: 0,
        stage_count: 11,
      },
    ],
    "/api/meetings/IS1001d/status": {
      meeting_id: "IS1001d",
      summary: {
        meeting_id: "IS1001d",
        has_raw_audio: true,
        has_staged_audio: false,
        has_artifacts: false,
        last_updated: "2026-02-28T12:00:00Z",
        config_digest: null,
        artifact_digest: null,
        offline_preflight_ok: null,
        determinism_risks: [],
        stages_complete: 0,
        stage_count: 11,
      },
      stages: [],
      artifact_count: 0,
      run_controls_enabled: false,
    },
    "/api/meetings/IS1001d/artifacts": [
      {
        name: "raw_audio",
        path: "/tmp/IS1001d.Mix-Headset.wav",
        relative_path: "data/rawa/ami/audio/IS1001d.Mix-Headset.wav",
        exists: true,
        kind: "audio",
        size_bytes: 1024,
        download_url: "/api/meetings/IS1001d/artifact/raw_audio/download",
        preview_url: "/api/meetings/IS1001d/artifact/raw_audio",
      },
      {
        name: "transcript_raw.json",
        path: "/tmp/transcript_raw.json",
        relative_path: "artifacts/ami/IS1001d/transcript_raw.json",
        exists: false,
        kind: "missing",
        size_bytes: null,
        download_url: "/api/meetings/IS1001d/artifact/transcript_raw.json/download",
        preview_url: "/api/meetings/IS1001d/artifact/transcript_raw.json",
      },
      {
        name: "transcript_normalized.json",
        path: "/tmp/transcript_normalized.json",
        relative_path: "artifacts/ami/IS1001d/transcript_normalized.json",
        exists: false,
        kind: "missing",
        size_bytes: null,
        download_url: "/api/meetings/IS1001d/artifact/transcript_normalized.json/download",
        preview_url: "/api/meetings/IS1001d/artifact/transcript_normalized.json",
      },
      {
        name: "transcript_chunks.jsonl",
        path: "/tmp/transcript_chunks.jsonl",
        relative_path: "artifacts/ami/IS1001d/transcript_chunks.jsonl",
        exists: false,
        kind: "missing",
        size_bytes: null,
        download_url: "/api/meetings/IS1001d/artifact/transcript_chunks.jsonl/download",
        preview_url: "/api/meetings/IS1001d/artifact/transcript_chunks.jsonl",
      },
    ],
    "/api/meetings/IS1001d/repro": {
      meeting_id: "IS1001d",
      config_digest: null,
      artifact_digest: null,
      determinism_risks: [],
    },
    "/api/meetings/IS1001d/speech": {
      meeting_id: "IS1001d",
      audio: {
        artifact: {
          name: "raw_audio",
          path: "/tmp/IS1001d.Mix-Headset.wav",
          relative_path: "data/rawa/ami/audio/IS1001d.Mix-Headset.wav",
          exists: true,
          kind: "audio",
          size_bytes: 1024,
          download_url: "/api/meetings/IS1001d/artifact/raw_audio/download",
          preview_url: "/api/meetings/IS1001d/artifact/raw_audio",
        },
        available: true,
      },
      vad_segments: [],
      diarization_segments: [],
      asr_segments: [],
    },
    "/api/eval/meeting/IS1001d": {
      meeting_id: "IS1001d",
      metrics: {},
      confidence: {},
      quality_checks: {},
    },
  });

  renderApp("/meetings/IS1001d");
  await waitFor(() => expect(screen.getByText("Artifact Explorer")).toBeInTheDocument());
  expect(screen.getByText("Coverage")).toBeInTheDocument();
  expect(screen.getByText("transcript_raw.json")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Transcript" }));
  await waitFor(() =>
    expect(
      screen.getByText(
        "Transcript artifacts are not available for this meeting. The pipeline may not have reached canonicalization or chunking.",
      ),
    ).toBeInTheDocument(),
  );
});
