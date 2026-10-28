import { decimal, methodLabel } from "@/lib/format";
import type { MethodRow } from "@/lib/types";

interface Props {
  rows: MethodRow[];
  primaryMethod: string;
}

/** The table is the accessible equivalent of the plot, and the audit trail. */
export function MethodTable({ rows, primaryMethod }: Props) {
  return (
    <div className="table-wrap">
      <table className="methods">
        <thead>
          <tr>
            <th scope="col">Method</th>
            <th scope="col">Estimate</th>
            <th scope="col" className="interval-column">
              95% interval
            </th>
            <th scope="col" className="width-column">
              Width
            </th>
            <th scope="col">Width vs unadjusted</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const primary = row.method === primaryMethod;
            const ratio = row.width_ratio;
            const fill = ratio === null ? 0 : Math.min(Math.max(ratio, 0), 1);
            const over = ratio !== null && ratio > 1.0005;
            return (
              <tr key={row.method} data-primary={primary}>
                <td>
                  {methodLabel(row.method)}
                  {primary ? (
                    <>
                      {" "}
                      <span className="tag tag-primary">primary</span>
                    </>
                  ) : null}
                </td>
                <td>{decimal(row.estimate)}</td>
                <td className="interval-cell interval-column">
                  {decimal(row.ci_low)} to {decimal(row.ci_high)}
                </td>
                <td className="width-column">{decimal(row.ci_width)}</td>
                <td>
                  {ratio === null ? "not available" : decimal(ratio, 3)}
                  <span className="bar" aria-hidden="true">
                    <span
                      className="bar-fill"
                      data-over={over}
                      style={{ width: `${Math.max(fill * 100, 1.5)}%` }}
                    />
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
