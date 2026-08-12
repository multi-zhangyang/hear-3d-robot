import { createHash } from "node:crypto";
import {
  mkdir,
  open,
  readFile,
  truncate
} from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import {
  createDenseHumanoidPolicyFrame,
  DenseHumanoidPolicyFrameSchema,
  DensePolicyRolloutReferenceSchema,
  type DenseHumanoidPolicyFrame,
  type DensePolicyRolloutReference
} from "../domain/humanoid-policy-rollout.js";
import type {
  HumanoidPolicyControlFrame,
  HumanoidPolicyFrameSink
} from "../world/humanoid/simulation.js";

const RUN_ID_PATTERN = /^[a-zA-Z0-9_-]+$/;

export interface DensePolicyExecutionExtent {
  frameCount: number;
  worldBeforeRevision: number;
  worldAfterRevision: number;
}

interface DensePolicyFrameIdentity {
  localFrameIndex: number;
  previousFrameSha256: string | null;
  frameSha256: string;
  callStepIndex: number;
  worldFrameBefore: number;
  worldFrameAfter: number;
}

interface DensePolicyCallState {
  path: string;
  frames: DensePolicyFrameIdentity[];
  byCallStep: Map<number, DensePolicyFrameIdentity>;
  byWorldFrame: Map<number, DensePolicyFrameIdentity>;
}

/**
 * A deployment-side, per-Skill-Call JSONL sink.
 *
 * Each record is synced before control returns to the world runtime. The writer
 * never owns a background process or a long-lived file handle, and serializes
 * calls so a single simulation cannot interleave its hash chain.
 */
export class DensePolicyRolloutWriter {
  readonly #rootDir: string;
  readonly #runId: string;
  readonly #states = new Map<string, DensePolicyCallState>();
  #tail: Promise<void> = Promise.resolve();

  constructor(input: { rootDir: string; runId: string }) {
    this.#rootDir = resolve(input.rootDir);
    this.#runId = requiredRunId(input.runId);
  }

  readonly recordFrame: HumanoidPolicyFrameSink = async (frame) => {
    const pending = this.#tail.then(() => this.#recordFrame(frame));
    this.#tail = pending.catch(() => undefined);
    return pending;
  };

  async flush(): Promise<void> {
    await this.#tail;
  }

  async #recordFrame(frame: HumanoidPolicyControlFrame): Promise<void> {
    const callId = frame.taskCommand.identity.callId;
    if (frame.taskCommand.identity.runtimeKind !== "semantic_skill") {
      throw new Error("Dense policy collection accepts only semantic Skill Calls");
    }
    const state = await this.#state(callId);
    const callStepIndex = frame.taskCommand.window.stepIndex;
    const worldFrame = frame.taskCommand.authority.worldFrame;
    const replay = state.byCallStep.get(callStepIndex)
      ?? state.byWorldFrame.get(worldFrame);
    if (replay) {
      if (replay.callStepIndex !== callStepIndex
        || replay.worldFrameBefore !== worldFrame) {
        throw new Error(
          `Dense policy replay identity diverged for Skill Call ${callId}`
        );
      }
      const candidate = createDenseHumanoidPolicyFrame({
        runId: this.#runId,
        localFrameIndex: replay.localFrameIndex,
        previousFrameSha256: replay.previousFrameSha256,
        frame
      });
      if (candidate.frame_sha256 !== replay.frameSha256) {
        throw new Error(
          `Dense policy replay payload diverged for Skill Call ${callId} `
          + `at step ${callStepIndex}`
        );
      }
      return;
    }
    const previous = state.frames.at(-1);
    if (previous && (callStepIndex <= previous.callStepIndex
      || worldFrame < previous.worldFrameAfter)) {
      throw new Error(
        `Dense policy frames moved backwards for Skill Call ${callId}`
      );
    }
    const record = createDenseHumanoidPolicyFrame({
      runId: this.#runId,
      localFrameIndex: state.frames.length,
      previousFrameSha256: previous?.frameSha256 ?? null,
      frame
    });
    await appendAndSync(state.path, `${JSON.stringify(record)}\n`);
    indexFrame(state, record);
  }

  async #state(callId: string): Promise<DensePolicyCallState> {
    const existing = this.#states.get(callId);
    if (existing) return existing;
    const path = densePolicyRolloutPath(this.#rootDir, this.#runId, callId);
    const frames = await loadAndRepairDensePolicyFrames(path, {
      runId: this.#runId,
      callId
    });
    const state: DensePolicyCallState = {
      path,
      frames: [],
      byCallStep: new Map(),
      byWorldFrame: new Map()
    };
    for (const frame of frames) indexFrame(state, frame);
    this.#states.set(callId, state);
    return state;
  }
}

export function densePolicyRolloutPath(
  rootDir: string,
  runId: string,
  callId: string
): string {
  const safeRunId = requiredRunId(runId);
  const callDigest = createHash("sha256").update(callId).digest("hex");
  return resolve(rootDir, safeRunId, `${callDigest}.jsonl`);
}

export async function loadDensePolicyRolloutReference(input: {
  rootDir: string;
  runId: string;
  callId: string;
  execution?: DensePolicyExecutionExtent | null;
}): Promise<DensePolicyRolloutReference | null> {
  const path = densePolicyRolloutPath(
    input.rootDir,
    input.runId,
    input.callId
  );
  const frames = await loadAndRepairDensePolicyFrames(path, {
    runId: input.runId,
    callId: input.callId
  });
  if (frames.length === 0) return null;
  const first = frames[0]!;
  const last = frames.at(-1)!;
  const execution = input.execution ?? null;
  const observationProtocols = new Set<string>();
  const actionProtocols = new Set<string>();
  let teacherFrameCount = 0;
  let pairedTeacherFrameCount = 0;
  let missingCallStepCount = 0;
  let missingWorldFrameCount = 0;
  for (let index = 0; index < frames.length; index += 1) {
    const frame = frames[index]!;
    if (frame.supervision.kind === "reference_teacher"
      || frame.supervision.kind === "paired_teacher") {
      teacherFrameCount += 1;
    }
    if (frame.supervision.kind === "paired_teacher") {
      pairedTeacherFrameCount += 1;
    }
    for (const component of frame.controller.inference?.components ?? []) {
      observationProtocols.add(component.observation.protocol);
      actionProtocols.add(component.action.protocol);
    }
    const previous = frames[index - 1];
    if (!previous) continue;
    missingCallStepCount += Math.max(
      0,
      frame.call_step_index - previous.call_step_index - 1
    );
    missingWorldFrameCount += Math.max(
      0,
      frame.world_frame_before - previous.world_frame_after
    );
  }
  const raw = await readFile(path);
  return DensePolicyRolloutReferenceSchema.parse({
    available: true,
    protocol: "hear-dense-policy-rollout-reference-v1",
    dataset_ref: `dense-policy-jsonl-v1:${input.runId}/${basename(path)}`,
    file_sha256: createHash("sha256").update(raw).digest("hex"),
    frame_count: frames.length,
    first_local_frame_index: 0,
    last_local_frame_index: last.local_frame_index,
    first_call_step_index: first.call_step_index,
    last_call_step_index: last.call_step_index,
    first_world_frame: first.world_frame_before,
    last_world_frame: last.world_frame_after,
    first_world_revision: first.world_revision_before,
    last_world_revision: last.world_revision_after,
    first_frame_sha256: first.frame_sha256,
    last_frame_sha256: last.frame_sha256,
    complete_from_window_start: first.call_step_index === 0
      && (!execution
        || first.world_revision_before === execution.worldBeforeRevision),
    complete_through_execution_end: execution !== null
      && last.world_revision_after === execution.worldAfterRevision
      && frames.length + missingWorldFrameCount === execution.frameCount,
    missing_call_step_count: missingCallStepCount,
    missing_world_frame_count: missingWorldFrameCount,
    teacher_frame_count: teacherFrameCount,
    paired_teacher_frame_count: pairedTeacherFrameCount,
    observation_protocols: [...observationProtocols].sort(compareCodePoints),
    action_protocols: [...actionProtocols].sort(compareCodePoints)
  });
}

async function loadAndRepairDensePolicyFrames(
  path: string,
  expected: { runId: string; callId: string }
): Promise<DenseHumanoidPolicyFrame[]> {
  let content: string;
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    if (isMissing(error)) return [];
    throw error;
  }
  if (content.length === 0) return [];
  if (!content.endsWith("\n")) {
    const lastNewline = content.lastIndexOf("\n");
    const tail = content.slice(lastNewline + 1);
    try {
      JSON.parse(tail);
      await appendAndSync(path, "\n");
      content += "\n";
    } catch {
      const validPrefix = content.slice(0, lastNewline + 1);
      await truncateAndSync(path, Buffer.byteLength(validPrefix, "utf8"));
      content = validPrefix;
    }
  }
  const records = content.slice(0, -1).split("\n");
  if (records.length === 1 && records[0] === "") return [];
  const frames: DenseHumanoidPolicyFrame[] = [];
  for (let index = 0; index < records.length; index += 1) {
    const line = records[index]!;
    if (!line.trim()) {
      throw new Error(`Dense policy rollout contains a blank record: ${path}`);
    }
    let value: unknown;
    try {
      value = JSON.parse(line) as unknown;
    } catch (error) {
      throw new Error(
        `Dense policy rollout record ${index} is not valid JSON: ${path}`,
        { cause: error }
      );
    }
    const frame = DenseHumanoidPolicyFrameSchema.parse(value);
    const previous = frames.at(-1);
    if (frame.run_id !== expected.runId || frame.call_id !== expected.callId) {
      throw new Error(`Dense policy rollout identity mismatch: ${path}`);
    }
    if (frame.local_frame_index !== index
      || frame.previous_frame_sha256 !== (previous?.frame_sha256 ?? null)) {
      throw new Error(`Dense policy rollout hash chain is discontinuous: ${path}`);
    }
    if (previous && (frame.call_step_index <= previous.call_step_index
      || frame.world_frame_before < previous.world_frame_after)) {
      throw new Error(`Dense policy rollout frame order is invalid: ${path}`);
    }
    frames.push(frame);
  }
  return frames;
}

function indexFrame(
  state: DensePolicyCallState,
  frame: DenseHumanoidPolicyFrame | DensePolicyFrameIdentity
): void {
  const identity: DensePolicyFrameIdentity = "frame_sha256" in frame
    ? {
        localFrameIndex: frame.local_frame_index,
        previousFrameSha256: frame.previous_frame_sha256,
        frameSha256: frame.frame_sha256,
        callStepIndex: frame.call_step_index,
        worldFrameBefore: frame.world_frame_before,
        worldFrameAfter: frame.world_frame_after
      }
    : frame;
  if (state.byCallStep.has(identity.callStepIndex)
    || state.byWorldFrame.has(identity.worldFrameBefore)) {
    throw new Error(`Dense policy rollout contains duplicate frame identity: ${state.path}`);
  }
  state.frames.push(identity);
  state.byCallStep.set(identity.callStepIndex, identity);
  state.byWorldFrame.set(identity.worldFrameBefore, identity);
}

async function appendAndSync(path: string, value: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const handle = await open(path, "a");
  try {
    await handle.writeFile(value, "utf8");
    await handle.sync();
  } finally {
    await handle.close();
  }
}

async function truncateAndSync(path: string, length: number): Promise<void> {
  await truncate(path, length);
  const handle = await open(path, "r+");
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
}

function requiredRunId(value: string): string {
  if (!RUN_ID_PATTERN.test(value)) {
    throw new Error("Dense policy rollout run ID is not filesystem-safe");
  }
  return value;
}

function isMissing(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
