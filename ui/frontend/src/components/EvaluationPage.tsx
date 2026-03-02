import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { formatNumber } from "../lib/format";
import type { EvalSummaryResponse } from "../lib/types";
import { EmptyState, Panel } from "./Primitives";

export function EvaluationPage() {
  const [data, setData] = useState<EvalSummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getEvalSummary().then(setData).catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) {
    return <EmptyState message={`Evaluation summary unavailable: ${error}`} />;
  }
  if (!data) {
    return <EmptyState message="Loading evaluation summary..." />;
  }

  return (
    <div className="space-y-6">
      <Panel title="Aggregate Evaluation" subtitle="WER, CER, cpWER, DER, and ROUGE metrics from local evaluation outputs">
        {Object.keys(data.aggregate_metrics).length === 0 ? (
          <EmptyState message="No evaluation metrics are available yet." />
        ) : (
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {Object.entries(data.aggregate_metrics).map(([key, value]) => (
              <div key={key} className="rounded-lg border border-border bg-surface/70 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-textSecondary">{key}</p>
                <p className="mt-2 text-2xl font-semibold text-textPrimary">
                  {typeof value === "number" ? formatNumber(value) : value}
                </p>
              </div>
            ))}
          </div>
        )}
      </Panel>

      <Panel title="Meeting-Level Scores" subtitle="Source rows from local evaluation outputs">
        {data.rows.length === 0 ? (
          <EmptyState message="No meeting-level evaluation rows were found." />
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="text-textSecondary">
                <tr>
                  {Object.keys(data.rows[0] ?? {}).map((column) => (
                    <th key={column} className="pb-3 font-medium">
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row) => (
                  <tr key={row.meeting_id} className="border-t border-border">
                    {Object.entries(row).map(([key, value]) => (
                      <td key={key} className="py-3">
                        {value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
