import { createHash, randomUUID } from "node:crypto";
import { readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import { loadWorkyardContactTrainingContract } from "./workyard-contact-contract.js";

const SHA256 = /^[a-f0-9]{64}$/u;
const REACH_DEPLOYMENT_PROTOCOL = "hear-whole-body-reach-policy-deployment-v3";
const REACH_GATE_PROTOCOL = "hear-typescript-mujoco-reach-deployment-gate-v1";
const REACH_OBSERVATION_PROTOCOL = "hear-workyard-whole-body-reach-observation-v5";
const REACH_ACTION_PROTOCOL = "bounded-whole-body-reach-mean";

interface ReachDeploymentReport {
  protocol?: string;
  deployment?: {
    protocol?: string;
    accepted?: boolean;
    controller_mode?: string;
    terminal_assistance_step_count?: number;
    minimum_support_margin_m?: number;
    maximum_foot_planar_displacement_m?: number;
    maximum_foot_slip_speed_m_s?: number;
    double_support_loss_rate_maximum?: number;
    no_foot_contact_rate_maximum?: number;
  };
  source?: {
    checkpoint_sha256?: string;
    phase_one_accepted?: boolean;
    hand_checkpoint_expansion_authorized?: boolean;
  };
  policy?: {
    file?: string;
    bytes?: number;
    sha256?: string;
    input?: string;
    input_size?: number;
    output?: string;
    output_size?: number;
    batch_dynamic?: boolean;
    gradient_parameter_count?: number;
  };
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const contractPath = resolve(options.contract);
  const repositoryRoot = resolve(dirname(contractPath), "..");
  const contract = await loadWorkyardContactTrainingContract(contractPath);
  const reachRoot = resolve(
    repositoryRoot,
    options.reachRoot ?? contract.qualified_inputs.reach_policy.root
  );
  const reachRootRelative = repositoryRelative(repositoryRoot, reachRoot, "Reach root");
  const jitPath = resolve(reachRoot, contract.qualified_inputs.reach_policy.jit);
  const reportPath = resolve(reachRoot, contract.qualified_inputs.reach_policy.report);
  const [jit, reportBytes, contractBytes] = await Promise.all([
    readFile(jitPath),
    readFile(reportPath),
    readFile(contractPath)
  ]);
  const report = JSON.parse(reportBytes.toString("utf8")) as ReachDeploymentReport;
  const jitSha256 = sha256(jit);
  const checkpointSha256 = report.source?.checkpoint_sha256;
  assertQualifiedReach(report, jit.byteLength, jitSha256,
    contract.qualified_inputs.reach_policy.jit);
  if (!checkpointSha256 || !SHA256.test(checkpointSha256)) {
    throw new Error("Qualified Reach report has no valid source checkpoint SHA256");
  }

  const preflightPath = resolve(
    repositoryRoot,
    options.preflightReport
      ?? `artifacts/training/workyard-contact-teacher/reach-${jitSha256}.json`
  );
  const preflightRelative = repositoryRelative(
    repositoryRoot, preflightPath, "Contact preflight report"
  );
  const raw = JSON.parse(contractBytes.toString("utf8")) as {
    qualified_inputs?: {
      reach_policy?: Record<string, unknown>;
      analytic_teacher_preflight?: Record<string, unknown>;
    };
  };
  const reach = raw.qualified_inputs?.reach_policy;
  const preflight = raw.qualified_inputs?.analytic_teacher_preflight;
  if (!reach || !preflight) {
    throw new Error("Contact contract has no qualified Reach/preflight inputs");
  }
  const changed = reach.root !== reachRootRelative
    || reach.jit_sha256 !== jitSha256
    || reach.source_checkpoint_sha256 !== checkpointSha256
    || preflight.report !== preflightRelative;
  reach.root = reachRootRelative;
  reach.jit_sha256 = jitSha256;
  reach.source_checkpoint_sha256 = checkpointSha256;
  preflight.report = preflightRelative;

  if (changed) {
    await atomicWriteJson(contractPath, raw);
    await loadWorkyardContactTrainingContract(contractPath);
  }
  process.stdout.write(`${JSON.stringify({
    protocol: "hear-workyard-contact-reach-pin-v1",
    changed,
    contract: repositoryRelative(repositoryRoot, contractPath, "Contact contract"),
    reach_root: reachRootRelative,
    reach_jit_sha256: jitSha256,
    source_checkpoint_sha256: checkpointSha256,
    required_preflight_report: preflightRelative,
    next: `pnpm teacher:workyard:contact:colab`
  }, null, 2)}\n`);
}

function assertQualifiedReach(
  report: ReachDeploymentReport,
  jitBytes: number,
  jitSha256: string,
  expectedFile: string
): void {
  const deployment = report.deployment;
  const source = report.source;
  const policy = report.policy;
  if (report.protocol !== REACH_DEPLOYMENT_PROTOCOL
    || deployment?.protocol !== REACH_GATE_PROTOCOL
    || deployment.accepted !== true
    || deployment.controller_mode !== "learned_policy_only"
    || deployment.terminal_assistance_step_count !== 0
    || !((deployment.minimum_support_margin_m ?? -Infinity) >= -0.04)
    || !((deployment.maximum_foot_planar_displacement_m ?? Infinity) <= 0.08)
    || !((deployment.maximum_foot_slip_speed_m_s ?? Infinity) <= 0.20)
    || !((deployment.double_support_loss_rate_maximum ?? Infinity) <= 0.10)
    || !((deployment.no_foot_contact_rate_maximum ?? Infinity) <= 0.01)
    || source?.phase_one_accepted !== true
    || source.hand_checkpoint_expansion_authorized !== true
    || policy?.file !== expectedFile
    || policy.bytes !== jitBytes
    || policy.sha256 !== jitSha256
    || policy.input !== REACH_OBSERVATION_PROTOCOL
    || policy.input_size !== 246
    || policy.output !== REACH_ACTION_PROTOCOL
    || policy.output_size !== 29
    || policy.batch_dynamic !== true
    || policy.gradient_parameter_count !== 0) {
    throw new Error(
      "Reach deployment is not qualified for immutable Contact training input"
    );
  }
}

async function atomicWriteJson(path: string, value: unknown): Promise<void> {
  const temporary = resolve(
    dirname(path), `.${randomUUID()}.workyard-contact-contract.tmp`
  );
  try {
    await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, {
      encoding: "utf8",
      flag: "wx"
    });
    await rename(temporary, path);
  } finally {
    await rm(temporary, { force: true });
  }
}

function repositoryRelative(root: string, path: string, label: string): string {
  const value = relative(root, path).replaceAll("\\", "/");
  if (!value || value === "." || value === ".." || value.startsWith("../")) {
    throw new Error(`${label} must remain inside the repository: ${path}`);
  }
  return value;
}

function sha256(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function parseArgs(values: string[]): {
  contract: string;
  reachRoot?: string;
  preflightReport?: string;
} {
  if (values[0] === "--") values = values.slice(1);
  const result: {
    contract: string;
    reachRoot?: string;
    preflightReport?: string;
  } = { contract: "training/workyard-contact-task-v1.json" };
  for (let index = 0; index < values.length; index += 2) {
    const name = values[index];
    const value = values[index + 1];
    if (!value) throw new Error(`Missing value for ${name ?? "argument"}`);
    if (name === "--contract") result.contract = value;
    else if (name === "--reach-root") result.reachRoot = value;
    else if (name === "--preflight-report") result.preflightReport = value;
    else throw new Error(`Unknown Contact pin option: ${name}`);
  }
  return result;
}

await main();
