import { createHash } from "node:crypto";
import { realpath, readFile } from "node:fs/promises";
import { isAbsolute, resolve } from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  HUMANOID_LEARNED_POLICY_CAPABILITIES
} from "../../domain/humanoid-policy.js";
import {
  HUMANOID_POLICY_OBSERVATION_FEATURES,
  type HumanoidControllerDescriptor,
  type HumanoidWholeBodyController,
  type HumanoidWholeBodyControllerFactory
} from "./whole-body-controller.js";

export const HUMANOID_CONTROLLER_MODULE_ENV = "HEAR_HUMANOID_CONTROLLER_MODULE";
const HUMANOID_CONTROLLER_MODULE_FACTORY =
  "createHumanoidWholeBodyController";

export interface HumanoidControllerModuleContext {
  readonly protocol: "hear-humanoid-controller-module-v1";
  readonly sourceSha256: string;
}

export interface HumanoidControllerSource {
  readonly sourceSha256: string;
  readonly controllerFactory: HumanoidWholeBodyControllerFactory;
}

export type HumanoidControllerModuleFactory = (
  context: HumanoidControllerModuleContext
) => HumanoidWholeBodyController | Promise<HumanoidWholeBodyController>;

export async function loadConfiguredHumanoidControllerSource(
  environment: NodeJS.ProcessEnv = process.env,
  baseDirectory = process.cwd()
): Promise<HumanoidControllerSource | undefined> {
  const specifier = environment[HUMANOID_CONTROLLER_MODULE_ENV]?.trim();
  if (!specifier) return undefined;
  return loadHumanoidControllerSource(specifier, baseDirectory);
}

export async function loadHumanoidControllerSource(
  specifier: string,
  baseDirectory = process.cwd()
): Promise<HumanoidControllerSource> {
  const request = specifier.trim();
  if (!request) throw new Error("Humanoid controller module specifier is empty");
  const entryPath = await resolveControllerEntry(request, baseDirectory);
  const sourceSha256 = createHash("sha256")
    .update(await readFile(entryPath))
    .digest("hex");
  const moduleUrl = pathToFileURL(entryPath);
  moduleUrl.searchParams.set("hear_controller_sha256", sourceSha256);
  const namespace: unknown = await import(moduleUrl.href);
  const moduleFactory = controllerModuleFactory(namespace);
  const instances = new WeakSet<object>();
  const context: HumanoidControllerModuleContext = Object.freeze({
    protocol: "hear-humanoid-controller-module-v1",
    sourceSha256
  });
  const controllerFactory: HumanoidWholeBodyControllerFactory = async () => {
    const controller = await moduleFactory(context);
    assertHumanoidWholeBodyController(controller);
    if (instances.has(controller)) {
      throw new Error(
        "Humanoid controller module factory returned a previously used stateful instance"
      );
    }
    instances.add(controller);
    return controller;
  };
  return Object.freeze({ sourceSha256, controllerFactory });
}

async function resolveControllerEntry(
  specifier: string,
  baseDirectory: string
): Promise<string> {
  const resolver = createRequire(
    pathToFileURL(resolve(baseDirectory, "hear-controller-resolver.cjs"))
  );
  let request = specifier;
  if (specifier.startsWith("file:")) {
    request = fileURLToPath(specifier);
  } else if (isPathSpecifier(specifier)) {
    request = isAbsolute(specifier) ? specifier : resolve(baseDirectory, specifier);
  }
  let resolvedEntry: string;
  try {
    resolvedEntry = resolver.resolve(request);
  } catch (error) {
    throw new Error(`Unable to resolve humanoid controller module: ${specifier}`, {
      cause: error
    });
  }
  return realpath(resolvedEntry);
}

function isPathSpecifier(specifier: string): boolean {
  return isAbsolute(specifier)
    || specifier.startsWith("./")
    || specifier.startsWith("../")
    || specifier.startsWith(".\\")
    || specifier.startsWith("..\\");
}

function controllerModuleFactory(namespace: unknown): HumanoidControllerModuleFactory {
  if (!isRecord(namespace)) throw missingFactoryError();
  const direct = namespace[HUMANOID_CONTROLLER_MODULE_FACTORY];
  if (typeof direct === "function") return direct as HumanoidControllerModuleFactory;
  const defaultExport = namespace.default;
  if (isRecord(defaultExport)) {
    const commonJsFactory = defaultExport[HUMANOID_CONTROLLER_MODULE_FACTORY];
    if (typeof commonJsFactory === "function") {
      return commonJsFactory as HumanoidControllerModuleFactory;
    }
  }
  throw missingFactoryError();
}

function missingFactoryError(): Error {
  return new Error(
    `Humanoid controller module must export ${HUMANOID_CONTROLLER_MODULE_FACTORY}(context)`
  );
}

function assertHumanoidWholeBodyController(
  value: unknown
): asserts value is HumanoidWholeBodyController {
  if (!isRecord(value)) {
    throw new Error("Humanoid controller module factory returned a non-object value");
  }
  for (const method of [
    "reset",
    "infer",
    "advanceHistory",
    "captureState",
    "restoreState",
    "dispose"
  ] as const) {
    if (typeof value[method] !== "function") {
      throw new Error(`Humanoid controller is missing method: ${method}`);
    }
  }
  assertControllerDescriptor(value.descriptor);
}

function assertControllerDescriptor(
  value: unknown
): asserts value is HumanoidControllerDescriptor {
  if (!isRecord(value)
    || value.protocol !== "humanoid-controller-v1"
    || value.actuation !== "joint_position_pd"
    || !isNonEmptyString(value.implementation)
    || !isPositiveFinite(value.controlStepSeconds)
    || !isPositiveFinite(value.physicsStepSeconds)) {
    throw new Error("Humanoid controller declares an invalid descriptor");
  }
  const ratio = value.controlStepSeconds / value.physicsStepSeconds;
  if (ratio < 1 || Math.abs(ratio - Math.round(ratio)) > 1e-9) {
    throw new Error("Humanoid controller timing must use an integer control-step ratio");
  }
  for (const optionalPositive of [
    value.commandResponseHorizonSeconds,
    value.minimumEffectivePlanarSpeedMetersPerSecond
  ]) {
    if (optionalPositive !== undefined && !isPositiveFinite(optionalPositive)) {
      throw new Error("Humanoid controller declares an invalid optional timing value");
    }
  }
  if (value.learnedPolicy !== undefined) {
    assertLearnedPolicyDescriptor(value.learnedPolicy);
  }
}

function assertLearnedPolicyDescriptor(value: unknown): void {
  if (!isRecord(value)
    || value.protocol !== "humanoid-learned-policy-v1"
    || !isNonEmptyString(value.runtime)
    || !isSpaceDescriptor(value.observationSpace)
    || !isSpaceDescriptor(value.actionSpace)
    || !isStringArrayFrom(value.capabilities, HUMANOID_LEARNED_POLICY_CAPABILITIES)) {
    throw new Error("Humanoid controller declares an invalid learned-policy descriptor");
  }
  if (value.observationFeatures !== undefined
    && !isStringArrayFrom(
      value.observationFeatures,
      HUMANOID_POLICY_OBSERVATION_FEATURES
    )) {
    throw new Error("Humanoid controller declares unsupported observation features");
  }
}

function isSpaceDescriptor(value: unknown): boolean {
  return isRecord(value)
    && isNonEmptyString(value.protocol)
    && typeof value.size === "number"
    && Number.isInteger(value.size)
    && value.size > 0;
}

function isStringArrayFrom(
  value: unknown,
  allowed: readonly string[]
): boolean {
  return Array.isArray(value)
    && value.every((entry) => typeof entry === "string" && allowed.includes(entry));
}

function isPositiveFinite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
