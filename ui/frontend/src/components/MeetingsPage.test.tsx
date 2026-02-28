import { screen, waitFor } from "@testing-library/react";
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

const meetingRows = [
  {
    meeting_id: "ES2005a",
    has_raw_audio: true,
    has_staged_audio: true,
    has_artifacts: true,
    last_updated: "2026-02-28T12:00:00Z",
    config_digest: "abc",
    artifact_digest: "def",
    offline_preflight_ok: true,
    determinism_risks: [],
    stages_complete: 10,
    stage_count: 11,
  },
];

test("meetings list renders mocked api data", async () => {
  installFetchMock({
    "/api/meetings": meetingRows,
    "/api/meetings/ES2005a/status": {
      meeting_id: "ES2005a",
      summary: meetingRows[0],
      stages: [],
      artifact_count: 0,
      run_controls_enabled: false,
    },
    "/api/meetings/ES2005a/artifacts": [],
    "/api/meetings/ES2005a/repro": {
      meeting_id: "ES2005a",
      config_digest: "abc",
      artifact_digest: "def",
      determinism_risks: [],
    },
    "/api/meetings/ES2005a/speech": {
      meeting_id: "ES2005a",
      audio: { artifact: {}, available: false },
      vad_segments: [],
      diarization_segments: [],
      asr_segments: [],
    },
    "/api/eval/meeting/ES2005a": {
      meeting_id: "ES2005a",
      metrics: {},
      confidence: {},
      quality_checks: {},
    },
  });
  renderApp("/meetings");
  await waitFor(() => expect(screen.getByText("ES2005a")).toBeInTheDocument());
});

test("meeting detail renders stage timeline from mocked status response", async () => {
  installFetchMock({
    "/api/meetings": meetingRows,
    "/api/meetings/ES2005a/status": {
      meeting_id: "ES2005a",
      summary: meetingRows[0],
      stages: [
        {
          name: "ASR",
          key: "asr",
          status: "success",
          runtime_sec: 1.23,
          artifacts: [],
          notes: [],
        },
      ],
      artifact_count: 4,
      run_controls_enabled: false,
    },
    "/api/meetings/ES2005a/artifacts": [],
    "/api/meetings/ES2005a/repro": {
      meeting_id: "ES2005a",
      config_digest: "abc",
      artifact_digest: "def",
      determinism_risks: [],
    },
    "/api/meetings/ES2005a/speech": {
      meeting_id: "ES2005a",
      audio: { artifact: {}, available: false },
      vad_segments: [],
      diarization_segments: [],
      asr_segments: [],
    },
    "/api/eval/meeting/ES2005a": {
      meeting_id: "ES2005a",
      metrics: {},
      confidence: {},
      quality_checks: {},
    },
    "/api/meetings/ES2005a/artifact/vad_segments.json": {
      meeting_id: "ES2005a",
      artifact: { name: "vad_segments.json" },
      content: [],
    },
    "/api/meetings/ES2005a/artifact/diarization_segments.json": {
      meeting_id: "ES2005a",
      artifact: { name: "diarization_segments.json" },
      content: [],
    },
    "/api/meetings/ES2005a/artifact/asr_segments.json": {
      meeting_id: "ES2005a",
      artifact: { name: "asr_segments.json" },
      content: [],
    },
  });
  renderApp("/meetings/ES2005a");
  await waitFor(() => expect(screen.getByText("Stage Timeline")).toBeInTheDocument());
  expect(screen.getByText("ASR")).toBeInTheDocument();
});

test("artifact tab renders json preview correctly", async () => {
  installFetchMock({
    "/api/meetings": meetingRows,
    "/api/meetings/ES2005a/status": {
      meeting_id: "ES2005a",
      summary: meetingRows[0],
      stages: [],
      artifact_count: 2,
      run_controls_enabled: false,
    },
    "/api/meetings/ES2005a/artifacts": [
      {
        name: "vad_segments.json",
        path: "/tmp/vad_segments.json",
        relative_path: "artifacts/ami/ES2005a/vad_segments.json",
        exists: true,
        kind: "json",
        size_bytes: 12,
        download_url: "/api/meetings/ES2005a/artifact/vad_segments.json/download",
        preview_url: "/api/meetings/ES2005a/artifact/vad_segments.json",
      },
      {
        name: "diarization_segments.json",
        path: "/tmp/diarization_segments.json",
        relative_path: "artifacts/ami/ES2005a/diarization_segments.json",
        exists: true,
        kind: "json",
        size_bytes: 12,
        download_url: "/api/meetings/ES2005a/artifact/diarization_segments.json/download",
        preview_url: "/api/meetings/ES2005a/artifact/diarization_segments.json",
      },
      {
        name: "asr_segments.json",
        path: "/tmp/asr_segments.json",
        relative_path: "artifacts/ami/ES2005a/asr_segments.json",
        exists: true,
        kind: "json",
        size_bytes: 12,
        download_url: "/api/meetings/ES2005a/artifact/asr_segments.json/download",
        preview_url: "/api/meetings/ES2005a/artifact/asr_segments.json",
      },
    ],
    "/api/meetings/ES2005a/repro": {
      meeting_id: "ES2005a",
      config_digest: "abc",
      artifact_digest: "def",
      determinism_risks: [],
    },
    "/api/meetings/ES2005a/speech": {
      meeting_id: "ES2005a",
      audio: {
        artifact: {
          name: "raw_audio",
          path: "/tmp/raw.wav",
          relative_path: "data/rawa/ami/audio/ES2005a.Mix-Headset.wav",
          exists: true,
          kind: "audio",
          size_bytes: 12,
          download_url: "/api/meetings/ES2005a/artifact/raw_audio/download",
          preview_url: "/api/meetings/ES2005a/artifact/raw_audio",
        },
        available: true,
      },
      vad_segments: [{ start: 0, end: 1, label: "speech" }],
      diarization_segments: [],
      asr_segments: [],
    },
    "/api/eval/meeting/ES2005a": {
      meeting_id: "ES2005a",
      metrics: {},
      confidence: {},
      quality_checks: {},
    },
  });
  renderApp("/meetings/ES2005a");
  await waitFor(() => expect(screen.getByText("VAD Segments")).toBeInTheDocument());
  await waitFor(() => expect(screen.getByText("speech")).toBeInTheDocument());
});

test("meeting run controls render when run controls are enabled", async () => {
  installFetchMock({
    "/api/meetings": meetingRows,
    "/api/meetings/ES2005a/status": {
      meeting_id: "ES2005a",
      summary: meetingRows[0],
      stages: [],
      artifact_count: 2,
      run_controls_enabled: true,
    },
    "/api/meetings/ES2005a/artifacts": [],
    "/api/meetings/ES2005a/repro": {
      meeting_id: "ES2005a",
      config_digest: "abc",
      artifact_digest: "def",
      determinism_risks: [],
    },
    "/api/meetings/ES2005a/speech": {
      meeting_id: "ES2005a",
      audio: { artifact: {}, available: false },
      vad_segments: [],
      diarization_segments: [],
      asr_segments: [],
    },
    "/api/eval/meeting/ES2005a": {
      meeting_id: "ES2005a",
      metrics: {},
      confidence: {},
      quality_checks: {},
    },
    "/api/configs": [{ name: "pipeline.nemo.llama.final_eval.yaml", path: "/tmp/pipeline.nemo.llama.final_eval.yaml", size_bytes: 100 }],
    "/api/meetings/ES2005a/runs": [],
  });
  renderApp("/meetings/ES2005a");
  await waitFor(() => expect(screen.getByText("Run And Validate")).toBeInTheDocument());
  expect(screen.getByText("Start")).toBeInTheDocument();
});
