import { lazy, Suspense, useEffect, useState } from "react";
import { api } from "../lib/api";
import { formatBytes } from "../lib/format";
import type { ConfigEntry, ConfigResponse } from "../lib/types";
import { EmptyState, Panel } from "./Primitives";

const MonacoEditor = lazy(() => import("@monaco-editor/react"));

export function ConfigurationPage() {
  const [configs, setConfigs] = useState<ConfigEntry[]>([]);
  const [selectedName, setSelectedName] = useState("");
  const [selectedConfig, setSelectedConfig] = useState<ConfigResponse | null>(null);
  const [isLoadingConfigs, setIsLoadingConfigs] = useState(true);
  const [isLoadingConfig, setIsLoadingConfig] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setIsLoadingConfigs(true);
    setError(null);
    api
      .getConfigs()
      .then((items) => {
        setConfigs(items);
        if (items[0]) {
          setSelectedName(items[0].name);
        }
      })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setIsLoadingConfigs(false));
  }, []);

  useEffect(() => {
    if (!selectedName) {
      return;
    }
    setIsLoadingConfig(true);
    setError(null);
    api
      .getConfig(selectedName)
      .then(setSelectedConfig)
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setIsLoadingConfig(false));
  }, [selectedName]);

  if (error) {
    return <EmptyState message={`Configuration view unavailable: ${error}`} />;
  }
  if (isLoadingConfigs) {
    return <EmptyState message="Loading configuration files..." />;
  }
  if (!configs.length) {
    return <EmptyState message="No configuration files found." />;
  }

  return (
    <div className="space-y-6">
      <Panel title="Configuration Profiles" subtitle="Read-only YAML viewer for local pipeline configs">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <select
            value={selectedName}
            onChange={(event) => setSelectedName(event.target.value)}
            className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-textPrimary outline-none ring-accent focus:ring-2 md:w-96"
          >
            {configs.map((config) => (
              <option key={config.name} value={config.name}>
                {config.name}
              </option>
            ))}
          </select>
          <div className="text-sm text-textSecondary">
            {configs.find((config) => config.name === selectedName)
              ? formatBytes(configs.find((config) => config.name === selectedName)?.size_bytes)
              : null}
          </div>
        </div>
        <p className="mt-3 text-sm text-textSecondary">
          {configs.length} configuration {configs.length === 1 ? "profile" : "profiles"} available locally.
        </p>
      </Panel>
      <Panel title={selectedName} subtitle="Editing is disabled in V1 read-only mode">
        <div className="overflow-hidden rounded-lg border border-border">
          {isLoadingConfig && !selectedConfig ? (
            <div className="min-h-[32rem] bg-background/80 p-4 text-sm text-textSecondary">Loading configuration...</div>
          ) : (
            <Suspense fallback={<div className="min-h-[32rem] bg-background/80 p-4 text-sm text-textSecondary">Loading editor...</div>}>
              <MonacoEditor
                height="32rem"
                defaultLanguage="yaml"
                theme="vs-dark"
                value={selectedConfig?.content ?? ""}
                options={{
                  readOnly: true,
                  minimap: { enabled: false },
                  lineNumbers: "on",
                  scrollBeyondLastLine: false,
                  wordWrap: "on",
                }}
              />
            </Suspense>
          )}
        </div>
      </Panel>
    </div>
  );
}
