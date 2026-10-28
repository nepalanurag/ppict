"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  INITIAL_CONTROLS,
  ProtocolPanel,
  type Controls,
  type DatasetView,
} from "@/components/ProtocolPanel";
import { Results } from "@/components/Results";
import type { ApiErrorBody, Report } from "@/lib/types";

interface RunError {
  message: string;
  detail?: string;
}

const DEMO_NAME = "simulated_trial.csv";

function keyOf(controls: Controls, file: File | null): string {
  return JSON.stringify([controls, file ? [file.name, file.size, file.lastModified] : "demo"]);
}

function optionalNumber(value: string): number | null {
  if (value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function validate(controls: Controls): string | null {
  const cost = optionalNumber(controls.cost);
  if (cost === null || Number.isNaN(cost) || cost < 0) {
    return "Cost per patient must be zero or more.";
  }
  const rate = optionalNumber(controls.rate);
  if (rate === null || Number.isNaN(rate) || rate <= 0) {
    return "Recruitment rate must be greater than zero.";
  }
  const r2 = optionalNumber(controls.r2);
  if (r2 !== null && (Number.isNaN(r2) || r2 < 0 || r2 >= 0.995)) {
    return "Prognostic R² must be between 0 and 0.99.";
  }
  const effect = optionalNumber(controls.effect);
  if (effect !== null && (Number.isNaN(effect) || effect <= 0)) {
    return "Treatment effect must be greater than zero.";
  }
  const size = optionalNumber(controls.trialSize);
  if (size !== null && (Number.isNaN(size) || size < 2)) {
    return "Planned trial size must be at least 2 patients.";
  }
  return null;
}

export function Calculator() {
  const [controls, setControls] = useState<Controls>(INITIAL_CONTROLS);
  const [file, setFile] = useState<File | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [runKey, setRunKey] = useState<string>("");
  const [pending, setPending] = useState(true);
  const [railError, setRailError] = useState<string | null>(null);
  const [error, setError] = useState<RunError | null>(null);
  const started = useRef(false);
  const latestRequest = useRef(0);

  const submit = useCallback(async (nextControls: Controls, nextFile: File | null) => {
    const invalid = validate(nextControls);
    if (invalid) {
      setRailError(invalid);
      return;
    }

    const requestId = latestRequest.current + 1;
    latestRequest.current = requestId;
    setRailError(null);
    setPending(true);

    const body = new FormData();
    if (nextFile) body.append("file", nextFile);
    body.append("cost", nextControls.cost);
    body.append("power", nextControls.power);
    body.append("patientsPerMonth", nextControls.rate);
    if (nextControls.r2.trim() !== "") body.append("r2", nextControls.r2);
    if (nextControls.effect.trim() !== "") body.append("effect", nextControls.effect);
    if (nextControls.trialSize.trim() !== "") body.append("trialSize", nextControls.trialSize);

    try {
      const response = await fetch("/api/analyze", { method: "POST", body, cache: "no-store" });
      const payload = (await response.json()) as Report | ApiErrorBody;
      // A newer run started while this one was in flight. Its result is the one
      // that matches the controls on screen, so drop this response.
      if (latestRequest.current !== requestId) return;
      if (!response.ok) {
        const failure = payload as ApiErrorBody;
        setError({
          message: failure.error || "The analysis did not run.",
          detail: failure.detail,
        });
        setReport(null);
        return;
      }
      setReport(payload as Report);
      setRunKey(keyOf(nextControls, nextFile));
      setError(null);
    } catch (caught) {
      if (latestRequest.current !== requestId) return;
      setError({
        message: "The calculator could not reach its analysis route.",
        detail: caught instanceof Error ? caught.message : String(caught),
      });
      setReport(null);
    } finally {
      if (latestRequest.current === requestId) setPending(false);
    }
  }, []);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    void submit(INITIAL_CONTROLS, null);
  }, [submit]);

  function change(patch: Partial<Controls>) {
    setControls((previous) => ({ ...previous, ...patch }));
    setRailError(null);
  }

  function changePower(value: string) {
    const next = { ...controls, power: value };
    setControls(next);
    void submit(next, file);
  }

  function pickFile(picked: File) {
    const next: Controls = { ...controls, r2: "", effect: "", trialSize: "" };
    setFile(picked);
    setControls(next);
    setReport(null);
    setError(null);
    void submit(next, picked);
  }

  function useExample() {
    const next: Controls = { ...controls, r2: "", effect: "", trialSize: "" };
    setFile(null);
    setControls(next);
    setReport(null);
    setError(null);
    void submit(next, null);
  }

  function download() {
    if (!report) return;
    const blob = new Blob([report.sap_tex], { type: "application/x-tex;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = report.sap_filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const stale = report !== null && keyOf(controls, file) !== runKey;

  const dataset: DatasetView = {
    name: file?.name ?? DEMO_NAME,
    kind: file ? "file" : "demo",
    nTrial: report?.analysis.dataset.n_trial ?? null,
    nExternal: report?.analysis.dataset.n_external ?? null,
    nDropped: report?.analysis.dataset.n_dropped ?? null,
    sha256: report?.analysis.dataset.sha256 ?? null,
  };

  return (
    <>
      <header className="masthead">
        <div>
          <h1 className="wordmark">ppict</h1>
          <p className="descriptor">
            Prediction-powered covariate adjustment for randomized trials. Every number below comes
            from the same Python estimator that the command line runs.
          </p>
        </div>
        <div className="masthead-meta">
          <div>Local analysis</div>
          <div>{report ? `ppict ${report.ppict_version}` : "ppict"}</div>
        </div>
      </header>

      <main className="layout">
        <ProtocolPanel
          controls={controls}
          measured={report?.analysis.measured ?? null}
          dataset={dataset}
          pending={pending}
          stale={stale}
          hasReport={report !== null}
          railError={railError}
          onChange={change}
          onPower={changePower}
          onFile={pickFile}
          onExample={useExample}
          onRun={() => void submit(controls, file)}
        />

        <section className="findings" aria-live="polite" aria-busy={pending}>
          {error ? (
            <div className="error" role="alert">
              <h2 className="error-title">The analysis did not run</h2>
              <p className="error-message">{error.message}</p>
              {error.detail ? <pre className="error-detail">{error.detail}</pre> : null}
              <button className="error-action" type="button" onClick={useExample}>
                Try the simulated example
              </button>
            </div>
          ) : report ? (
            <Results
              report={report}
              stale={stale}
              pending={pending}
              sourceLabel={file ? "uploaded file" : "simulated example"}
              onDownload={download}
            />
          ) : (
            <div className="working">
              <span className="working-rule" aria-hidden="true" />
              Running the estimator on the loaded data.
            </div>
          )}
        </section>
      </main>
    </>
  );
}
