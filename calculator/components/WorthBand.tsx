import { count, money, months, percent } from "@/lib/format";
import type { Plan } from "@/lib/types";

/**
 * Three design projections. Every one of them is model-based, which the note
 * below the band says plainly; none is an observed saving.
 */
export function WorthBand({ plan }: { plan: Plan }) {
  if (plan.patients_saved === null) return null;

  return (
    <>
      <div className="worth">
        <div className="worth-cell">
          <span className="worth-value">{count(plan.patients_saved)}</span>
          <span className="worth-label">fewer patients</span>
          <span className="worth-detail">
            {count(plan.n_required_unadjusted)} to {count(plan.n_required_adjusted)} at{" "}
            {percent(plan.inputs.power, 0)} power
          </span>
        </div>
        <div className="worth-cell">
          <span className="worth-value">{money(plan.cost_saved)}</span>
          <span className="worth-label">trial cost avoided</span>
          <span className="worth-detail">
            {money(plan.inputs.cost_per_patient)} per patient, held constant
          </span>
        </div>
        <div className="worth-cell">
          <span className="worth-value">{months(plan.months_saved)}</span>
          <span className="worth-label">recruitment time saved</span>
          <span className="worth-detail">
            at {count(plan.inputs.patients_per_month)} patients per month
          </span>
        </div>
      </div>
      <p className="worth-note">
        Projected under the variance model above, not observed savings. The patient saving is the
        difference between the unadjusted and adjusted requirements at the stated power, so it is
        attributable to the adjustment rather than to the size of the plan. The interval narrowing
        at the top is measured on the loaded file.
      </p>
    </>
  );
}
