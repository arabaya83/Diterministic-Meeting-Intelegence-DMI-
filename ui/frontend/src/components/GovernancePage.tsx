import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { EmptyState, Panel, StatusBadge } from "./Primitives";
import type { GovernanceResponse } from "../lib/types";

export function GovernancePage() {
  const [data, setData] = useState<GovernanceResponse | null>(null);
  const [evidence, setEvidence] = useState<Array<Record<string, unknown>>>([]);
  const [mlflowConfigured, setMlflowConfigured] = useState(false);
  const [mlflowRuns, setMlflowRuns] = useState<Array<Record<string, unknown>>>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.getGovernance(), api.getEvidenceBundles(), api.getMlflowRuns()])
      .then(([governance, evidenceResponse, mlflowResponse]) => {
        setData(governance);
        setEvidence(evidenceResponse.items);
        setMlflowConfigured(Boolean(mlflowResponse.configured));
        setMlflowRuns(mlflowResponse.items);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) {
    return <EmptyState message={`Governance view unavailable: ${error}`} />;
  }
  if (!data) {
    return <EmptyState message="Loading governance state..." />;
  }

  return (
    <div className="space-y-6">
      <Panel title="Evidence Bundles" subtitle="Acceptance and evidence exports discovered on local disk">
        {evidence.length === 0 ? (
          <EmptyState message="No evidence bundles found." />
        ) : (
          <div className="space-y-2">
            {evidence.map((bundle, index) => (
              <div key={index} className="rounded-lg border border-border bg-surface/70 p-3 text-sm">
                <p className="font-medium text-textPrimary">{String(bundle.name ?? "bundle")}</p>
                <p className="mt-1 text-textSecondary">{String(bundle.path ?? "")}</p>
              </div>
            ))}
          </div>
        )}
      </Panel>

      <Panel title="MLflow" subtitle="Local run metadata if configured">
        <div className="mb-4">
          <StatusBadge state={mlflowConfigured ? "success" : "warn"} label={mlflowConfigured ? "Configured" : "Not configured"} />
        </div>
        {mlflowConfigured ? (
          <div className="space-y-2">
            {mlflowRuns.map((run, index) => (
              <div key={index} className="rounded-lg border border-border bg-surface/70 p-3 text-sm text-textPrimary">
                {String(run.path ?? "")}
              </div>
            ))}
          </div>
        ) : (
          <EmptyState message="MLflow is not configured for this repository." />
        )}
      </Panel>
    </div>
  );
}
