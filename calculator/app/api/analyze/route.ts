import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { EXPECTED_CONTRACT, type Report } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;

// The calculator runs from this checkout, so the paths below are runtime state,
// not build assets. The turbopackIgnore markers stop the bundler from tracing the
// whole project into the route just because it cannot resolve them statically.

/** Walk up from the app directory until the Python package is found. */
function findRepoRoot(): string {
  let dir = process.cwd();
  for (let depth = 0; depth < 4; depth += 1) {
    if (existsSync(/* turbopackIgnore: true */ path.join(dir, "src", "ppict", "cli.py"))) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  throw new Error("Could not find src/ppict/cli.py above this directory.");
}

function resolvePython(root: string): string {
  const configured = process.env.PPICT_PYTHON?.trim();
  if (configured) return configured;

  const candidates = [
    path.join(root, ".venv", "bin", "python"),
    path.join(root, ".venv", "Scripts", "python.exe"),
  ];
  for (const candidate of candidates) {
    if (existsSync(/* turbopackIgnore: true */ candidate)) return candidate;
  }
  return "python3";
}

interface RunResult {
  stdout: string;
  stderr: string;
  code: number;
  spawnError?: string;
}

function run(command: string, args: string[], cwd: string): Promise<RunResult> {
  return new Promise((resolve) => {
    execFile(
      command,
      args,
      { cwd, timeout: 60_000, maxBuffer: 16 * 1024 * 1024, env: process.env },
      (error, stdout, stderr) => {
        const rawCode = (error as NodeJS.ErrnoException | null)?.code;
        const numeric = typeof rawCode === "number";
        resolve({
          stdout: String(stdout ?? ""),
          stderr: String(stderr ?? ""),
          code: numeric ? (rawCode as unknown as number) : error ? 1 : 0,
          spawnError: error && !numeric ? String(rawCode ?? error.message) : undefined,
        });
      },
    );
  });
}

/** The package prints `error: ...` for input problems; prefer that line over a traceback. */
function errorMessage(stderr: string): string {
  const lines = stderr.split("\n").map((line) => line.trim()).filter(Boolean);
  const explicit = [...lines].reverse().find((line) => line.startsWith("error: "));
  if (explicit) return explicit.replace(/^error:\s*/, "");
  return lines.length ? lines[lines.length - 1] : "The estimator failed without a message.";
}

function tail(text: string, lines = 24): string {
  return text.split("\n").slice(-lines).join("\n").trim();
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

export async function POST(request: Request): Promise<Response> {
  let root: string;
  try {
    root = findRepoRoot();
  } catch (error) {
    return json(
      {
        error: "The calculator could not find the ppict package.",
        detail: error instanceof Error ? error.message : String(error),
      },
      500,
    );
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return json({ error: "The request was not readable as a form submission." }, 400);
  }

  const args = ["-m", "ppict.cli", "analyze"];
  const flags: [string, string][] = [
    ["cost", "--cost"],
    ["power", "--power"],
    ["r2", "--r2"],
    ["effect", "--effect"],
    ["trialSize", "--trial-size"],
    ["patientsPerMonth", "--patients-per-month"],
  ];

  for (const [field, flag] of flags) {
    const raw = form.get(field);
    if (typeof raw !== "string" || raw.trim() === "") continue;
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      return json({ error: `The value for ${flag.replace(/^--/, "")} is not a number.` }, 400);
    }
    if (flag === "--trial-size") {
      if (value < 2) return json({ error: "The planned trial size must be at least 2." }, 400);
      args.push(flag, String(Math.round(value)));
    } else {
      args.push(flag, String(value));
    }
  }

  let workDir: string | null = null;
  try {
    const upload = form.get("file");
    let csvPath: string;

    if (upload instanceof File) {
      if (upload.size === 0) {
        return json({ error: "The uploaded file is empty." }, 400);
      }
      if (upload.size > MAX_UPLOAD_BYTES) {
        return json({ error: "The uploaded file is larger than 5 MB." }, 413);
      }
      workDir = await mkdtemp(path.join(os.tmpdir(), "ppict-calculator-"));
      csvPath = path.join(workDir, "trial.csv");
      await writeFile(csvPath, Buffer.from(await upload.arrayBuffer()));
    } else {
      // Relative to the repository root, so the recorded command reads the way a
      // person would type it. The child process runs with that root as its cwd.
      const demo = path.join("calculator", "data", "simulated_trial.csv");
      if (!existsSync(/* turbopackIgnore: true */ path.join(root, demo))) {
        return json(
          {
            error: "The example dataset is missing.",
            detail: "Run: ./.venv/bin/python tools/make_demo_trial.py",
          },
          500,
        );
      }
      csvPath = demo;
    }

    args.push("--csv", csvPath);
    const python = resolvePython(root);
    const result = await run(python, args, root);

    if (result.spawnError) {
      return json(
        {
          error: "The Python interpreter could not be started.",
          detail: `Tried ${python} (${result.spawnError}). Set PPICT_PYTHON to the interpreter that has ppict installed.`,
        },
        503,
      );
    }

    if (result.code !== 0) {
      // The package prints one clean `error:` line for input problems. Show the
      // traceback only when there is one worth reading.
      const clean = result.stderr.trimStart().startsWith("error: ");
      return json(
        { error: errorMessage(result.stderr), detail: clean ? undefined : tail(result.stderr) },
        400,
      );
    }

    let report: Report;
    try {
      report = JSON.parse(result.stdout) as Report;
    } catch {
      return json(
        {
          error: "The estimator returned something that was not JSON.",
          detail: tail(result.stdout) || tail(result.stderr),
        },
        502,
      );
    }

    if (report.contract !== EXPECTED_CONTRACT) {
      return json(
        {
          error: "The calculator and the Python package disagree about the report format.",
          detail: `Expected contract ${EXPECTED_CONTRACT}, received ${String(report.contract)}. Reinstall the package and reload.`,
        },
        502,
      );
    }

    return json(report);
  } catch (error) {
    return json(
      {
        error: "The calculator could not run the analysis.",
        detail: error instanceof Error ? error.message : String(error),
      },
      500,
    );
  } finally {
    if (workDir) {
      await rm(workDir, { recursive: true, force: true }).catch(() => undefined);
    }
  }
}

export async function GET(): Promise<Response> {
  return json({ error: "Use POST to run an analysis." }, 405);
}
