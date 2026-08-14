import { createHash } from "node:crypto";
import { realpath, readFile } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";
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
const BUNDLED_MJLAB_G1_CONTROLLER_SOURCE =
  "hear-bundled-mjlab-g1-velocity-controller-v1";
const HUMANOID_CONTROLLER_MODULE_FACTORY =
  "createHumanoidWholeBodyController";
const HUMANOID_CONTROLLER_MODULE_ASSETS = "humanoidControllerAssets";

export interface HumanoidControllerModuleAsset {
  readonly id: string;
  readonly sha256: string;
  readonly bytes: Uint8Array;
}

export interface HumanoidControllerModuleContext {
  readonly protocol: "hear-humanoid-controller-module-v1";
  readonly sourceSha256: string;
  readonly assets: readonly HumanoidControllerModuleAsset[];
}

export interface HumanoidControllerSource {
  readonly sourceSha256: string;
  readonly controllerFactory: HumanoidWholeBodyControllerFactory;
}

export type HumanoidControllerModuleFactory = (
  context: HumanoidControllerModuleContext
) => HumanoidWholeBodyController | Promise<HumanoidWholeBodyController>;

interface HumanoidControllerAssetDeclaration {
  readonly id: string;
  readonly path: string | URL;
}

export async function loadConfiguredHumanoidControllerSource(
  environment: NodeJS.ProcessEnv = process.env,
  baseDirectory = process.cwd()
): Promise<HumanoidControllerSource> {
  const specifier = environment[HUMANOID_CONTROLLER_MODULE_ENV]?.trim();
  if (!specifier) return loadBundledMjlabG1ControllerSource(environment);
  return loadHumanoidControllerSource(specifier, baseDirectory);
}

async function loadBundledMjlabG1ControllerSource(
  environment: NodeJS.ProcessEnv
): Promise<HumanoidControllerSource> {
  const namespace: unknown = await import(
    "../../controllers/mjlab-g1-velocity-module.js"
  );
  const assets = await loadControllerAssets(
    namespace,
    fileURLToPath(import.meta.url),
    environment
  );
  return createControllerSource(
    namespace,
    controllerSourceSha256(
      sha256(BUNDLED_MJLAB_G1_CONTROLLER_SOURCE),
      assets
    ),
    assets
  );
}

export async function loadHumanoidControllerSource(
  specifier: string,
  baseDirectory = process.cwd()
): Promise<HumanoidControllerSource> {
  const request = specifier.trim();
  if (!request) throw new Error("Humanoid controller module specifier is empty");
  const entryPath = await resolveControllerEntry(request, baseDirectory);
  const entryBytes = await readFile(entryPath);
  const entrySha256 = sha256(entryBytes);
  const moduleUrl = pathToFileURL(entryPath);
  moduleUrl.searchParams.set("hear_controller_sha256", entrySha256);
  const namespace: unknown = await import(moduleUrl.href);
  const assets = await loadControllerAssets(namespace, entryPath);
  const sourceSha256 = controllerSourceSha256(entrySha256, assets);
  return createControllerSource(namespace, sourceSha256, assets);
}

function createControllerSource(
  namespace: unknown,
  sourceSha256: string,
  assets: readonly HumanoidControllerModuleAsset[]
): HumanoidControllerSource {
  const moduleFactory = controllerModuleFactory(namespace);
  const instances = new WeakSet<object>();
  const controllerFactory: HumanoidWholeBodyControllerFactory = async () => {
    const context: HumanoidControllerModuleContext = Object.freeze({
      protocol: "hear-humanoid-controller-module-v1",
      sourceSha256,
      assets: Object.freeze(assets.map((asset) => Object.freeze({
        id: asset.id,
        sha256: asset.sha256,
        bytes: asset.bytes.slice()
      })))
    });
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

async function loadControllerAssets(
  namespace: unknown,
  entryPath: string,
  environment?: NodeJS.ProcessEnv
): Promise<HumanoidControllerModuleAsset[]> {
  const assetExport = moduleExport(namespace, HUMANOID_CONTROLLER_MODULE_ASSETS);
  if (assetExport === undefined) return [];
  const exported: unknown = typeof assetExport === "function"
    ? await assetExport(...(environment ? [environment] : []))
    : assetExport;
  if (!Array.isArray(exported)) {
    throw new Error("Humanoid controller assets must be an array");
  }
  const declarations = exported.map((value, index) => (
    assetDeclaration(value, index)
  ));
  const ids = declarations.map(({ id }) => id);
  if (new Set(ids).size !== ids.length) {
    throw new Error("Humanoid controller asset identifiers must be unique");
  }
  const assets = await Promise.all(declarations.map(async ({ id, path }) => {
    const resolvedPath = await realpath(resolveAssetPath(path, entryPath));
    const bytes = await readFile(resolvedPath);
    return Object.freeze({ id, sha256: sha256(bytes), bytes });
  }));
  return assets.sort((left, right) => left.id.localeCompare(right.id));
}

function assetDeclaration(
  value: unknown,
  index: number
): HumanoidControllerAssetDeclaration {
  if (!isRecord(value)
    || !isNonEmptyString(value.id)
    || !/^[a-z0-9][a-z0-9._-]*$/.test(value.id)
    || !(isNonEmptyString(value.path) || value.path instanceof URL)) {
    throw new Error(`Humanoid controller asset ${index} is invalid`);
  }
  return { id: value.id, path: value.path };
}

function resolveAssetPath(path: string | URL, entryPath: string): string {
  if (path instanceof URL) {
    if (path.protocol !== "file:") {
      throw new Error("Humanoid controller assets must use local files");
    }
    return fileURLToPath(path);
  }
  if (path.startsWith("file:")) return fileURLToPath(path);
  return isAbsolute(path) ? path : resolve(dirname(entryPath), path);
}

function controllerSourceSha256(
  entrySha256: string,
  assets: readonly HumanoidControllerModuleAsset[]
): string {
  if (assets.length === 0) return entrySha256;
  return sha256(JSON.stringify({
    protocol: "hear-humanoid-controller-source-v2",
    entry_sha256: entrySha256,
    assets: assets.map(({ id, sha256: assetSha256, bytes }) => ({
      id,
      sha256: assetSha256,
      bytes: bytes.byteLength
    }))
  }));
}

function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
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
  const factory = moduleExport(namespace, HUMANOID_CONTROLLER_MODULE_FACTORY);
  if (typeof factory === "function") return factory as HumanoidControllerModuleFactory;
  throw missingFactoryError();
}

function moduleExport(namespace: unknown, name: string): unknown {
  if (!isRecord(namespace)) return undefined;
  const direct = namespace[name];
  if (direct !== undefined) return direct;
  const defaultExport = namespace.default;
  return isRecord(defaultExport) ? defaultExport[name] : undefined;
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
    value.minimumEffectivePlanarSpeedMetersPerSecond,
    value.minimumEffectiveYawSpeedRadiansPerSecond
  ]) {
    if (optionalPositive !== undefined && !isPositiveFinite(optionalPositive)) {
      throw new Error("Humanoid controller declares an invalid optional timing value");
    }
  }
  if (value.learnedPolicy !== undefined) {
    assertLearnedPolicyDescriptor(value.learnedPolicy);
  }
  if (value.capabilityRouting !== undefined) {
    assertCapabilityRoutingDescriptor(value.capabilityRouting, value.implementation);
    if (value.learnedPolicy === undefined
      || (isRecord(value.learnedPolicy)
        && Array.isArray(value.learnedPolicy.capabilities)
        && value.learnedPolicy.capabilities.includes("joint_reference_tracking"))) {
      throw new Error(
        "Humanoid capability routing requires a primary learned policy without reference tracking"
      );
    }
  }
}

function assertCapabilityRoutingDescriptor(
  value: unknown,
  primaryImplementation: string
): void {
  if (!isRecord(value)
    || value.protocol !== "humanoid-controller-capability-routing-v1"
    || (value.strategy !== "declared_capabilities"
      && value.strategy !== "capability_evidence")
    || !isRecord(value.fallback)
    || value.fallback.mode !== "reference_control"
    || !isNonEmptyString(value.fallback.implementation)
    || value.fallback.implementation === primaryImplementation) {
    throw new Error("Humanoid controller declares invalid capability routing");
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
