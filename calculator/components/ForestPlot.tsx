import { decimal, methodLabel } from "@/lib/format";
import type { MethodRow } from "@/lib/types";

const WIDTH = 900;
const LEFT = 196;
const RIGHT = 18;
const TOP = 18;
const ROW = 34;

function formatTick(value: number, step: number): string {
  if (Math.abs(value) < step / 1000) return "0";
  const digits = step < 0.1 ? 2 : 1;
  return value.toFixed(digits).replace(/\.?0+$/, "");
}

function tickStep(span: number): number {
  const candidates = [0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1];
  return candidates.find((step) => span / step <= 6) ?? 1;
}

interface Props {
  rows: MethodRow[];
  primaryMethod: string;
}

/**
 * Interval plot. Drawn from the estimator's numbers only: no smoothing, no
 * uncertainty the package did not report.
 */
export function ForestPlot({ rows, primaryMethod }: Props) {
  if (rows.length === 0) return null;

  const plotWidth = WIDTH - LEFT - RIGHT;
  let domainLow = Math.min(0, ...rows.map((row) => row.ci_low));
  let domainHigh = Math.max(0, ...rows.map((row) => row.ci_high));
  if (!Number.isFinite(domainLow) || !Number.isFinite(domainHigh) || domainHigh <= domainLow) {
    domainLow = -1;
    domainHigh = 1;
  }
  const pad = Math.max((domainHigh - domainLow) * 0.06, 0.01);
  const x0 = domainLow - pad;
  const x1 = domainHigh + pad;
  const sx = (value: number) => LEFT + ((value - x0) / (x1 - x0)) * plotWidth;

  const axisY = TOP + rows.length * ROW + 8;
  const height = axisY + 34;

  const step = tickStep(x1 - x0);
  const ticks: number[] = [];
  for (let value = Math.ceil(x0 / step) * step; value <= x1 + step / 1000 && ticks.length < 12; value += step) {
    ticks.push(Number(value.toFixed(6)));
  }

  const description = rows
    .map(
      (row) =>
        `${methodLabel(row.method)}: ${decimal(row.estimate)}, 95 percent interval ${decimal(
          row.ci_low,
        )} to ${decimal(row.ci_high)}`,
    )
    .join(". ");

  return (
    <svg
      className="plot"
      viewBox={`0 0 ${WIDTH} ${height}`}
      role="img"
      aria-label={`${description}. The dashed line marks no effect.`}
    >
      <title>Point estimates and 95 percent intervals</title>

      {rows.map((row, index) => {
        const y = TOP + index * ROW + ROW / 2;
        const primary = row.method === primaryMethod;
        const cap = 6;
        const lowX = sx(row.ci_low);
        const highX = sx(row.ci_high);
        const pointX = sx(row.estimate);
        return (
          <g key={row.method}>
            {primary ? (
              <rect className="wash" x={LEFT - 14} y={y - ROW / 2 + 2} width={plotWidth + 28} height={ROW - 4} />
            ) : null}
            <line className="track" x1={LEFT} x2={LEFT + plotWidth} y1={y} y2={y} />
            <text className={primary ? "label label-primary" : "label"} x={0} y={y + 4}>
              {methodLabel(row.method)}
            </text>
            <line
              className={primary ? "interval interval-primary" : "interval"}
              x1={lowX}
              x2={highX}
              y1={y}
              y2={y}
            />
            <line
              className={primary ? "cap cap-primary" : "cap"}
              x1={lowX}
              x2={lowX}
              y1={y - cap}
              y2={y + cap}
            />
            <line
              className={primary ? "cap cap-primary" : "cap"}
              x1={highX}
              x2={highX}
              y1={y - cap}
              y2={y + cap}
            />
            <circle className={primary ? "point point-primary" : "point"} cx={pointX} cy={y} r={primary ? 4.2 : 3.2} />
          </g>
        );
      })}

      <line className="null" x1={sx(0)} x2={sx(0)} y1={TOP - 6} y2={axisY} />
      <line className="axis" x1={LEFT} x2={LEFT + plotWidth} y1={axisY} y2={axisY} />
      {ticks.map((tick) => (
        <g key={tick}>
          <line className="axis" x1={sx(tick)} x2={sx(tick)} y1={axisY} y2={axisY + 4} />
          <text className="tick" x={sx(tick)} y={axisY + 17} textAnchor="middle">
            {formatTick(tick, step)}
          </text>
        </g>
      ))}
    </svg>
  );
}
