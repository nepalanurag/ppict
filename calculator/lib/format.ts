/** Formatting only. No statistics happens on the client. */

const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const integer = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function decimal(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "not available";
  return value.toFixed(digits);
}

export function money(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "not available";
  return usd.format(value);
}

export function count(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "not available";
  return integer.format(Math.round(value));
}

export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "not available";
  return `${(100 * value).toFixed(digits)}%`;
}

export function months(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "not available";
  return `${value.toFixed(1)} months`;
}

export function shortHash(hash: string, length = 12): string {
  return hash ? hash.slice(0, length) : "not recorded";
}

/**
 * Describe the interval width against the unadjusted analysis without claiming
 * a direction that the numbers do not show.
 */
export function narrowingPhrase(ratio: number | null): string {
  if (ratio === null || !Number.isFinite(ratio)) return "The interval comparison is not available.";
  const change = 1 - ratio;
  if (Math.abs(change) < 0.005) return "The interval is the same width as the unadjusted analysis.";
  if (change > 0) return `${percent(change)} narrower than the unadjusted analysis on the same patients.`;
  return `${percent(-change)} wider than the unadjusted analysis on the same patients.`;
}

const METHOD_LABELS: Record<string, string> = {
  unadjusted: "Unadjusted",
  ppi_single: "PPI++, tuned",
  ancova: "ANCOVA on the score",
  ppi_per_arm: "PPI++, per arm",
};

export function methodLabel(method: string): string {
  return METHOD_LABELS[method] ?? method;
}

