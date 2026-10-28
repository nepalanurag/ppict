import { Assumptions } from "@/components/Assumptions";
import { ForestPlot } from "@/components/ForestPlot";
import { MethodTable } from "@/components/MethodTable";
import { WorthBand } from "@/components/WorthBand";
import { count, decimal, methodLabel, narrowingPhrase } from "@/lib/format";
import type { Report } from "@/lib/types";

interface Props {
  report: Report;
  stale: boolean;
  pending: boolean;
  sourceLabel: string;
  onDownload: () => void;
}

export function Results({ report, stale, pending, sourceLabel, onDownload }: Props) {
  const methods = report.analysis.methods;
  const primary =
    methods.find((row) => row.method === report.analysis.primary_method) ?? methods[0];

  if (!primary) {
    return (
      <div className="error" role="alert">
        <h2 className="error-title">The estimator returned no methods</h2>
        <p className="error-message">
          The report parsed but contained no estimates. Reinstall the package and try again.
        </p>
      </div>
    );
  }

  const generated = report.generated_at.replace("T", " ").slice(0, 19);
  const exportBlocked = stale || pending;

  return (
    <div className="results" data-pending={pending}>
      <p className="hero-label">Adjusted treatment effect</p>
      <span className="hero-value">{decimal(primary.estimate)}</span>
      <p className="hero-ci">
        95% interval {decimal(primary.ci_low)} to {decimal(primary.ci_high)}
      </p>
      <p className="hero-finding">
        Measured on this file: {narrowingPhrase(report.analysis.primary_width_ratio)}
      </p>

      <div className="hero-meta">
        <div>
          Tuning value <span>{primary.lam === null ? "not applicable" : decimal(primary.lam, 3)}</span>
        </div>
        <div>
          Interval width <span>{decimal(primary.ci_width)}</span>
        </div>
        <div>
          Patients <span>{count(report.analysis.dataset.n_trial)}</span>
        </div>
        <div>
          Source <span>{sourceLabel}</span>
        </div>
      </div>

      <h2 className="section-title">What the narrowing is worth</h2>
      <WorthBand plan={report.plan} />
      {report.plan.notes.map((note) => (
        <p className="caution" key={note}>
          {note}
        </p>
      ))}

      <h2 className="section-title">Intervals on the same patients</h2>
      <ForestPlot rows={methods} primaryMethod={report.analysis.primary_method} />
      <p className="caption">
        Each bar is a 95% interval and each dot is the point estimate, on the same{" "}
        {count(report.analysis.dataset.n_trial)} patients. The dashed line marks no effect. ANCOVA
        adjusts for the same score and is shown because the two should agree.
      </p>

      <h2 className="section-title">Every method, same rows</h2>
      <MethodTable rows={methods} primaryMethod={report.analysis.primary_method} />
      {report.analysis.unavailable.map((item) => (
        <p className="caption" key={item.method}>
          {methodLabel(item.method)}: {item.reason}.
        </p>
      ))}

      <h2 className="section-title">What the projection assumes</h2>
      <Assumptions report={report} />

      <div className="export">
        <button className="export-button" type="button" onClick={onDownload} disabled={exportBlocked}>
          Download the SAP appendix (.tex)
        </button>
        <p className="export-note">
          {exportBlocked
            ? "Recompute first, so the appendix matches the inputs on screen."
            : `A complete LaTeX document for this analysis: the pre-specified model, the tuning rule, the design table and the exact command. Dated ${generated.slice(0, 10)}. It does not contain the observed effect.`}
        </p>
      </div>

      <dl className="provenance">
        <dt>Command</dt>
        <dd>{report.command}</dd>
        <dt>Report generated</dt>
        <dd>{generated}</dd>
        <dt>Input digest</dt>
        <dd>{report.analysis.dataset.sha256 || "not recorded"}</dd>
      </dl>
    </div>
  );
}
