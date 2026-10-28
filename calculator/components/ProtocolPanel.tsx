"use client";

import { useState, type DragEvent, type ReactNode } from "react";

import { count, decimal, shortHash } from "@/lib/format";
import type { Measured } from "@/lib/types";

export interface Controls {
  cost: string;
  power: string;
  trialSize: string;
  r2: string;
  effect: string;
  rate: string;
}

export const INITIAL_CONTROLS: Controls = {
  cost: "4200",
  power: "0.8",
  trialSize: "",
  r2: "",
  effect: "",
  rate: "25",
};

export interface DatasetView {
  name: string;
  kind: "demo" | "file";
  nTrial: number | null;
  nExternal: number | null;
  nDropped: number | null;
  sha256: string | null;
}

interface Props {
  controls: Controls;
  measured: Measured | null;
  dataset: DatasetView;
  pending: boolean;
  stale: boolean;
  hasReport: boolean;
  railError: string | null;
  onChange: (patch: Partial<Controls>) => void;
  onPower: (value: string) => void;
  onFile: (file: File) => void;
  onExample: () => void;
  onRun: () => void;
}

interface FieldProps {
  id: string;
  label: string;
  value: string;
  placeholder?: string;
  prefix?: string;
  hint: ReactNode;
  onChange: (value: string) => void;
}

function Field({ id, label, value, placeholder, prefix, hint, onChange }: FieldProps) {
  return (
    <label className="field" htmlFor={id}>
      <span className="field-label">{label}</span>
      <span className="field-input">
        {prefix ? <span className="field-prefix">{prefix}</span> : null}
        <input
          id={id}
          type="text"
          inputMode="decimal"
          autoComplete="off"
          spellCheck={false}
          value={value}
          placeholder={placeholder}
          onChange={(event) => onChange(event.target.value)}
        />
      </span>
      <span className="field-hint">{hint}</span>
    </label>
  );
}

export function ProtocolPanel({
  controls,
  measured,
  dataset,
  pending,
  stale,
  hasReport,
  railError,
  onChange,
  onPower,
  onFile,
  onExample,
  onRun,
}: Props) {
  const [dragging, setDragging] = useState(false);

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) onFile(file);
  }

  const measuredR2 = measured?.score_r2_control ?? null;
  const measuredEffect = measured?.effect_unadjusted ?? null;

  const datasetMeta =
    dataset.nTrial === null
      ? "Counts appear after a successful run."
      : `${count(dataset.nTrial)} trial patients, ${count(dataset.nExternal)} external${
          dataset.nDropped ? `, ${count(dataset.nDropped)} incomplete rows dropped` : ""
        }`;

  return (
    <aside className="rail" aria-label="Protocol inputs">
      <h2 className="rail-title">Protocol</h2>
      <p className="rail-intro">
        Edit a value and recompute. Numbers marked measured come from the loaded file; everything
        else is an assumption.
      </p>

      <div
        className="dataset"
        data-dragging={dragging}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <div className="dataset-head">
          <span className="dataset-name">{dataset.name}</span>
          {dataset.kind === "demo" ? <span className="tag tag-simulated">simulated</span> : null}
        </div>
        <p className="dataset-meta">
          {datasetMeta}
          {dataset.sha256 ? `. Digest ${shortHash(dataset.sha256)}` : ""}
        </p>
        <div className="dataset-actions">
          <label className="file-pick">
            Choose a CSV
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) onFile(file);
                event.target.value = "";
              }}
            />
          </label>
          {dataset.kind === "file" ? (
            <button className="text-button" type="button" onClick={onExample}>
              Use the example
            </button>
          ) : null}
        </div>
      </div>

      <div className="fields">
        <Field
          id="cost"
          label="Cost per patient"
          prefix="$"
          value={controls.cost}
          onChange={(value) => onChange({ cost: value })}
          hint="Assumption. It scales the projected cost saving."
        />

        <div className="field">
          <span className="field-label" id="power-label">
            Target power
          </span>
          <div className="segmented" role="group" aria-labelledby="power-label">
            <button
              type="button"
              aria-pressed={controls.power === "0.8"}
              onClick={() => onPower("0.8")}
            >
              80%
            </button>
            <button
              type="button"
              aria-pressed={controls.power === "0.9"}
              onClick={() => onPower("0.9")}
            >
              90%
            </button>
          </div>
          <span className="field-hint">Assumption. Higher power needs more patients.</span>
        </div>

        <Field
          id="r2"
          label="Prognostic R²"
          value={controls.r2}
          placeholder={measuredR2 === null ? undefined : decimal(measuredR2, 4)}
          onChange={(value) => onChange({ r2: value })}
          hint={
            controls.r2.trim() === "" ? (
              measuredR2 === null ? (
                "Measured from the loaded file."
              ) : (
                <>
                  Measured <em>{decimal(measuredR2, 4)}</em> in the control arm. Type a value to
                  test another assumption.
                </>
              )
            ) : (
              "Your assumption. It drives the cost and time projection, not the measured interval."
            )
          }
        />

        <Field
          id="effect"
          label="Treatment effect"
          value={controls.effect}
          placeholder={measuredEffect === null ? undefined : decimal(measuredEffect, 4)}
          onChange={(value) => onChange({ effect: value })}
          hint={
            controls.effect.trim() === "" ? (
              measuredEffect === null ? (
                "Measured from the loaded file."
              ) : (
                <>
                  Measured <em>{decimal(measuredEffect, 4)}</em> from the unadjusted analysis.
                </>
              )
            ) : (
              "Your assumption, used for the sample-size projection."
            )
          }
        />

        <Field
          id="trialSize"
          label="Planned trial size"
          value={controls.trialSize}
          placeholder={dataset.nTrial === null ? undefined : count(dataset.nTrial)}
          onChange={(value) => onChange({ trialSize: value })}
          hint={
            controls.trialSize.trim() === ""
              ? dataset.nTrial === null
                ? "Defaults to the size of the loaded file."
                : `From the dataset: ${count(dataset.nTrial)} patients.`
              : "Your assumption. Used for the information-equivalence line and the power check."
          }
        />

        <Field
          id="rate"
          label="Recruitment rate"
          value={controls.rate}
          onChange={(value) => onChange({ rate: value })}
          hint="Assumption, in patients per month. Used only for the time projection."
        />
      </div>

      <button className="run" type="button" onClick={onRun} disabled={pending}>
        {pending ? "Running the estimator" : hasReport ? "Recompute" : "Run the analysis"}
      </button>

      {railError ? (
        <p className="caution" role="alert">
          {railError}
        </p>
      ) : null}

      {stale && !railError ? (
        <p className="stale">Inputs changed. Recompute to update the numbers.</p>
      ) : null}
    </aside>
  );
}
