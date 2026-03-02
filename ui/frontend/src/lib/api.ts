import type {
  ArtifactEntry,
  ArtifactPreview,
  ConfigEntry,
  ConfigResponse,
  DashboardResponse,
  EvalSummaryResponse,
  GovernanceListResponse,
  GovernanceResponse,
  MeetingExtractionResponse,
  MeetingEvalResponse,
  MeetingListItem,
  MeetingReproResponse,
  MeetingRunEntry,
  MeetingSpeechResponse,
  MeetingStatusResponse,
  MeetingSummaryResponse,
  MeetingTranscriptResponse,
  RunMode,
  RunStatusResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  baseUrl: API_BASE,
  getDashboard: () => request<DashboardResponse>("/api/dashboard"),
  getMeetings: () => request<MeetingListItem[]>("/api/meetings"),
  getMeetingStatus: (meetingId: string) =>
    request<MeetingStatusResponse>(`/api/meetings/${meetingId}/status`),
  getArtifacts: (meetingId: string) =>
    request<ArtifactEntry[]>(`/api/meetings/${meetingId}/artifacts`),
  getMeetingSpeech: (meetingId: string) =>
    request<MeetingSpeechResponse>(`/api/meetings/${meetingId}/speech`),
  getMeetingTranscript: (meetingId: string) =>
    request<MeetingTranscriptResponse>(`/api/meetings/${meetingId}/transcript`),
  getMeetingSummaryTab: (meetingId: string) =>
    request<MeetingSummaryResponse>(`/api/meetings/${meetingId}/summary`),
  getMeetingExtraction: (meetingId: string) =>
    request<MeetingExtractionResponse>(`/api/meetings/${meetingId}/extraction`),
  getArtifactPreview: (meetingId: string, name: string) =>
    request<ArtifactPreview>(`/api/meetings/${meetingId}/artifact/${name}`),
  getEvalSummary: () => request<EvalSummaryResponse>("/api/eval/summary"),
  getMeetingEval: (meetingId: string) =>
    request<MeetingEvalResponse>(`/api/eval/meeting/${meetingId}`),
  getMeetingRepro: (meetingId: string) =>
    request<MeetingReproResponse>(`/api/meetings/${meetingId}/repro`),
  getRuns: () => request<MeetingRunEntry[]>("/api/runs"),
  getMeetingRuns: (meetingId: string) =>
    request<MeetingRunEntry[]>(`/api/meetings/${meetingId}/runs`),
  getConfigs: () => request<ConfigEntry[]>("/api/configs"),
  getConfig: (name: string) => request<ConfigResponse>(`/api/configs/${name}`),
  getGovernance: () => request<GovernanceResponse>("/api/governance"),
  getEvidenceBundles: () => request<GovernanceListResponse>("/api/governance/evidence-bundles"),
  getMlflowRuns: () => request<GovernanceListResponse>("/api/governance/mlflow/runs"),
  createRun: async (payload: { meeting_id?: string; meeting_ids?: string[]; config: string; mode: RunMode }) => {
    const response = await fetch(`${API_BASE}/api/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }
    return response.json() as Promise<RunStatusResponse>;
  },
  getRun: (runId: string) => request<RunStatusResponse>(`/api/runs/${runId}`),
  cancelRun: async (runId: string) => {
    const response = await fetch(`${API_BASE}/api/runs/${runId}/cancel`, { method: "POST" });
    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }
    return response.json() as Promise<RunStatusResponse>;
  },
  runWebSocketUrl: (runId: string) => {
    const url = new URL(API_BASE);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = `/api/runs/ws/${runId}`;
    return url.toString();
  },
  artifactDownloadUrl: (meetingId: string, name: string) =>
    `${API_BASE}/api/meetings/${meetingId}/artifact/${name}/download`,
};
