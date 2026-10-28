import { count, decimal, money, percent } from "@/lib/format";
import type { Report, SourceTag } from "@/lib/types";

interface Row {
  term: string;
  value: string;
  source?: SourceTag;
  context: string;
  mono?: boolean;
}

export function Assumptions({ report }: { report: Report }) {
  const { inputs, sources } = report.plan;
  const measured = report.analysis.measured;

  const r2Context =
    sources.r2 === "measured"
      ? "squared correlation in the control arm"
      : measured.score_r2_control === null
        ? "no prognostic signal measured; used as zero"
        : "your value, in place of the measured one";

  const rows: Row[] = [
    {
      term: "Treatment effect",
      value: decimal(inputs.effect, 4),
      source: sources.effect,
      context: sources.effect === "measured" ? "from the unadjusted estimate" : "your value",
    },
    {
      term: "Outcome SD",
      value: decimal(inputs.outcome_sd, 4),
      source: sources.outcome_sd,
      context: sources.outcome_sd === "measured" ? "pooled across arms" : "your value",
    },
    {
      term: "Prognostic R²",
      value: decimal(inputs.r2, 4),
      source: sources.r2,
      context: r2Context,
    },
    {
      term: "Target power",
      value: percent(inputs.power, 0),
      source: "assumed",
      context: "for the sample-size projection",
    },
    {
      term: "Cost per patient",
      value: money(inputs.cost_per_patient),
      source: "assumed",
      context: "held constant in the projection",
    },
    {
      term: "Recruitment",
      value: `${count(inputs.patients_per_month)} per month`,
      source: "assumed",
      context: "used only for the time projection",
    },
    {
      term: "Planned trial size",
      value: count(inputs.n_total),
      source: "assumed",
      context: `total patients; equal to ${count(report.plan.equivalent_unadjusted_n)} under an unadjusted analysis`,
    },
    {
      term: "Variance model",
      value: "4σ² (1 − ρ²)",
      context: "balanced arms, equal arm variances",
    },
  ];

  return (
    <dl className="assumptions">
      {rows.map((row) => (
        <div key={row.term} style={{ display: "contents" }}>
          <dt>{row.term}</dt>
          <dd>
            <span className={row.mono ? "value" : undefined}>{row.value}</span>
            {row.source ? (
              <>
                {" "}
                <span className={`tag tag-${row.source}`}>{row.source}</span>
              </>
            ) : null}{" "}
            <span className="context">{row.context}</span>
          </dd>
        </div>
      ))}
    </dl>
  );
}
