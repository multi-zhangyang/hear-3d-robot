import { createHash, randomUUID } from "node:crypto";
import {
  mkdir,
  readFile,
  realpath,
  rename,
  rm,
  unlink,
  writeFile
} from "node:fs/promises";
import { basename, dirname, isAbsolute, relative, resolve } from "node:path";
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
const KNOWN_HUMANOID_CONTROLLER_MODULES = [
  "hear/controllers/yahmp",
  "hear/controllers/mjlab-g1-velocity",
  "hear/controllers/workyard-reach",
  "hear/controllers/workyard-contact"
] as const;
const BUNDLED_YAHMP_CONTROLLER_SOURCE =
  "hear-bundled-yahmp-recovery-controller-v2";
const BUNDLED_WORKYARD_CONTROLLER_SOURCE =
  "hear-bundled-workyard-whole-body-contact-controller-v3";
const HUMANOID_CONTROLLER_MODULE_FACTORY =
  "createHumanoidWholeBodyController";
const HUMANOID_CONTROLLER_MODULE_ASSETS = "humanoidControllerAssets";
const CONTROLLER_SOURCE_ARCHIVE_DIRECTORY = "controller-source";
const CONTROLLER_SOURCE_ARCHIVE_PROTOCOL =
  "hear-humanoid-controller-source-archive-v1";
const SHA256_PATTERN = /^[a-f0-9]{64}$/;

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
  readonly archive?: HumanoidControllerSourceArchive;
}

interface HumanoidControllerSourceArchive {
  readonly entryBytes: Uint8Array;
  readonly entryFileName: string;
  readonly entryIdentitySha256: string;
  readonly runtimeDirectory: string;
  readonly assets: readonly LoadedHumanoidControllerModuleAsset[];
}

interface PersistedHumanoidControllerSourceArchive {
  readonly protocol: typeof CONTROLLER_SOURCE_ARCHIVE_PROTOCOL;
  readonly source_sha256: string;
  readonly entry_file: string;
  readonly entry_file_name: string;
  readonly entry_sha256: string;
  readonly entry_identity_sha256: string;
  readonly runtime_directory: string;
  readonly assets: readonly {
    id: string;
    file: string;
    sha256: string;
    bytes: number;
    source_identity: boolean;
  }[];
}

export type HumanoidControllerModuleFactory = (
  context: HumanoidControllerModuleContext
) => HumanoidWholeBodyController | Promise<HumanoidWholeBodyController>;

interface HumanoidControllerAssetDeclaration {
  readonly id: string;
  readonly path: string | URL;
  /** Metadata may be loaded by the module without changing executable identity. */
  readonly sourceIdentity: boolean;
}

interface LoadedHumanoidControllerModuleAsset
  extends HumanoidControllerModuleAsset {
  readonly sourceIdentity: boolean;
}

export async function loadConfiguredHumanoidControllerSource(
  environment: NodeJS.ProcessEnv = process.env,
  baseDirectory = process.cwd()
): Promise<HumanoidControllerSource> {
  const specifier = environment[HUMANOID_CONTROLLER_MODULE_ENV]?.trim();
  if (!specifier) return loadBundledYahmpControllerSource(environment);
  return loadHumanoidControllerSource(specifier, baseDirectory);
}

export async function findKnownHumanoidControllerSource(
  sourceSha256: string,
  environment: NodeJS.ProcessEnv = process.env,
  baseDirectory = process.cwd()
): Promise<HumanoidControllerSource | undefined> {
  for (const specifier of KNOWN_HUMANOID_CONTROLLER_MODULES) {
    try {
      const source = await loadHumanoidControllerSource(specifier, baseDirectory);
      if (source.sourceSha256 === sourceSha256) return source;
    } catch {
      continue;
    }
  }
  for (const loadBundled of [
    loadBundledYahmpControllerSource,
    loadBundledWorkyardControllerSource
  ]) {
    try {
      const bundled = await loadBundled(environment);
      if (bundled.sourceSha256 === sourceSha256) return bundled;
    } catch {
      continue;
    }
  }
  return undefined;
}

async function loadBundledYahmpControllerSource(
  environment: NodeJS.ProcessEnv
): Promise<HumanoidControllerSource> {
  const entryPath = await resolveControllerEntry(
    "hear/controllers/yahmp",
    process.cwd()
  );
  const entryBytes = await readFile(entryPath);
  const namespace: unknown = await import(pathToFileURL(entryPath).href);
  const assets = await loadControllerAssets(
    namespace,
    entryPath,
    environment
  );
  const entryIdentitySha256 = sha256(BUNDLED_YAHMP_CONTROLLER_SOURCE);
  return createControllerSource(
    namespace,
    controllerSourceSha256(entryIdentitySha256, assets),
    assets,
    controllerSourceArchive({
      entryPath,
      entryBytes,
      entryIdentitySha256,
      assets,
      baseDirectory: process.cwd()
    })
  );
}

async function loadBundledWorkyardControllerSource(
  environment: NodeJS.ProcessEnv
): Promise<HumanoidControllerSource> {
  const entryPath = await resolveControllerEntry(
    "hear/controllers/workyard-contact",
    process.cwd()
  );
  const entryBytes = await readFile(entryPath);
  const namespace: unknown = await import(pathToFileURL(entryPath).href);
  const assets = await loadControllerAssets(
    namespace,
    entryPath,
    environment
  );
  const entryIdentitySha256 = sha256(BUNDLED_WORKYARD_CONTROLLER_SOURCE);
  return createControllerSource(
    namespace,
    controllerSourceSha256(entryIdentitySha256, assets),
    assets,
    controllerSourceArchive({
      entryPath,
      entryBytes,
      entryIdentitySha256,
      assets,
      baseDirectory: process.cwd()
    })
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
  return createControllerSource(
    namespace,
    sourceSha256,
    assets,
    controllerSourceArchive({
      entryPath,
      entryBytes,
      entryIdentitySha256: entrySha256,
      assets,
      baseDirectory
    })
  );
}

export async function persistHumanoidControllerSourceArchive(
  runDirectory: string,
  source: HumanoidControllerSource
): Promise<void> {
  const archive = source.archive;
  if (!archive) return;
  const target = resolve(runDirectory, CONTROLLER_SOURCE_ARCHIVE_DIRECTORY);
  const staging = resolve(
    runDirectory,
    `.${CONTROLLER_SOURCE_ARCHIVE_DIRECTORY}-${randomUUID()}`
  );
  const entryFile = "entry.bin";
  const persistedAssets = archive.assets.map((asset, index) => ({
    id: asset.id,
    file: `asset-${index}-${asset.id}.bin`,
    sha256: asset.sha256,
    bytes: asset.bytes.byteLength,
    source_identity: asset.sourceIdentity
  }));
  const manifest: PersistedHumanoidControllerSourceArchive = {
    protocol: CONTROLLER_SOURCE_ARCHIVE_PROTOCOL,
    source_sha256: source.sourceSha256,
    entry_file: entryFile,
    entry_file_name: archive.entryFileName,
    entry_sha256: sha256(archive.entryBytes),
    entry_identity_sha256: archive.entryIdentitySha256,
    runtime_directory: archive.runtimeDirectory,
    assets: persistedAssets
  };
  try {
    await mkdir(staging, { recursive: false });
    await Promise.all([
      writeFile(resolve(staging, entryFile), archive.entryBytes),
      ...persistedAssets.map((persisted, index) => writeFile(
        resolve(staging, persisted.file),
        archive.assets[index]!.bytes
      ))
    ]);
    // The manifest is the archive commit record. Write it only after every
    // content-addressed byte payload has reached the staging directory.
    await writeFile(
      resolve(staging, "manifest.json"),
      `${JSON.stringify(manifest, null, 2)}\n`,
      "utf8"
    );
    await rename(staging, target);
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
}

export async function loadArchivedHumanoidControllerSource(
  runDirectory: string,
  expectedSourceSha256: string
): Promise<HumanoidControllerSource | undefined> {
  const archiveDirectory = resolve(
    runDirectory,
    CONTROLLER_SOURCE_ARCHIVE_DIRECTORY
  );
  let rawManifest: string;
  try {
    rawManifest = await readFile(resolve(archiveDirectory, "manifest.json"), "utf8");
  } catch (error) {
    if (isRecord(error) && error.code === "ENOENT") return undefined;
    throw error;
  }
  const manifest = persistedControllerSourceArchive(JSON.parse(rawManifest));
  if (manifest.source_sha256 !== expectedSourceSha256) {
    throw new Error("Humanoid controller archive does not match the Run identity");
  }
  const entryBytes = await readArchivedBytes(
    archiveDirectory,
    manifest.entry_file,
    manifest.entry_sha256
  );
  const assets = await Promise.all(manifest.assets.map(async (asset) => {
    const bytes = await readArchivedBytes(
      archiveDirectory,
      asset.file,
      asset.sha256
    );
    if (bytes.byteLength !== asset.bytes) {
      throw new Error(`Archived humanoid controller asset has wrong size: ${asset.id}`);
    }
    return Object.freeze({
      id: asset.id,
      sha256: asset.sha256,
      bytes,
      sourceIdentity: asset.source_identity
    });
  }));
  const reconstructedSourceSha256 = controllerSourceSha256(
    manifest.entry_identity_sha256,
    assets
  );
  if (reconstructedSourceSha256 !== expectedSourceSha256) {
    throw new Error("Humanoid controller archive failed source identity reconstruction");
  }
  const runtimeDirectory = safeRuntimeDirectory(
    process.cwd(),
    manifest.runtime_directory
  );
  const materializedEntry = resolve(
    runtimeDirectory,
    `.hear-${expectedSourceSha256}-${randomUUID()}-${manifest.entry_file_name}`
  );
  await writeFile(materializedEntry, entryBytes, { flag: "wx" });
  try {
    const moduleUrl = pathToFileURL(materializedEntry);
    moduleUrl.searchParams.set("hear_controller_sha256", expectedSourceSha256);
    const namespace: unknown = await import(moduleUrl.href);
    return createControllerSource(
      namespace,
      expectedSourceSha256,
      assets,
      controllerSourceArchive({
        entryPath: materializedEntry,
        entryBytes,
        entryIdentitySha256: manifest.entry_identity_sha256,
        assets,
        baseDirectory: process.cwd(),
        entryFileName: manifest.entry_file_name,
        runtimeDirectory: manifest.runtime_directory
      })
    );
  } finally {
    await unlink(materializedEntry).catch(() => undefined);
  }
}

function createControllerSource(
  namespace: unknown,
  sourceSha256: string,
  assets: readonly LoadedHumanoidControllerModuleAsset[],
  archive?: HumanoidControllerSourceArchive
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
  return Object.freeze({
    sourceSha256,
    controllerFactory,
    ...(archive ? { archive } : {})
  });
}

function controllerSourceArchive(input: {
  entryPath: string;
  entryBytes: Uint8Array;
  entryIdentitySha256: string;
  assets: readonly LoadedHumanoidControllerModuleAsset[];
  baseDirectory: string;
  entryFileName?: string;
  runtimeDirectory?: string;
}): HumanoidControllerSourceArchive | undefined {
  const runtimeDirectory = input.runtimeDirectory ?? relative(
    resolve(input.baseDirectory),
    dirname(input.entryPath)
  );
  if (!safeRelativePath(runtimeDirectory)) return undefined;
  return Object.freeze({
    entryBytes: input.entryBytes.slice(),
    entryFileName: input.entryFileName ?? basename(input.entryPath),
    entryIdentitySha256: input.entryIdentitySha256,
    runtimeDirectory,
    assets: Object.freeze(input.assets.map((asset) => Object.freeze({
      id: asset.id,
      sha256: asset.sha256,
      bytes: asset.bytes.slice(),
      sourceIdentity: asset.sourceIdentity
    })))
  });
}

async function loadControllerAssets(
  namespace: unknown,
  entryPath: string,
  environment?: NodeJS.ProcessEnv
): Promise<LoadedHumanoidControllerModuleAsset[]> {
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
  const assets = await Promise.all(declarations.map(async ({
    id,
    path,
    sourceIdentity
  }) => {
    const resolvedPath = await realpath(resolveAssetPath(path, entryPath));
    const bytes = await readFile(resolvedPath);
    return Object.freeze({
      id,
      sha256: sha256(bytes),
      bytes,
      sourceIdentity
    });
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
    || !(isNonEmptyString(value.path) || value.path instanceof URL)
    || (value.sourceIdentity !== undefined
      && typeof value.sourceIdentity !== "boolean")) {
    throw new Error(`Humanoid controller asset ${index} is invalid`);
  }
  return {
    id: value.id,
    path: value.path,
    sourceIdentity: value.sourceIdentity ?? true
  };
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
  assets: readonly LoadedHumanoidControllerModuleAsset[]
): string {
  const identityAssets = assets.filter(({ sourceIdentity }) => sourceIdentity);
  if (identityAssets.length === 0) return entrySha256;
  return sha256(JSON.stringify({
    protocol: "hear-humanoid-controller-source-v2",
    entry_sha256: entrySha256,
    assets: identityAssets.map(({ id, sha256: assetSha256, bytes }) => ({
      id,
      sha256: assetSha256,
      bytes: bytes.byteLength
    }))
  }));
}

function persistedControllerSourceArchive(
  value: unknown
): PersistedHumanoidControllerSourceArchive {
  if (!isRecord(value)
    || value.protocol !== CONTROLLER_SOURCE_ARCHIVE_PROTOCOL
    || !isSha256(value.source_sha256)
    || !isArchiveFileName(value.entry_file)
    || !isControllerEntryFileName(value.entry_file_name)
    || !isSha256(value.entry_sha256)
    || !isSha256(value.entry_identity_sha256)
    || typeof value.runtime_directory !== "string"
    || !safeRelativePath(value.runtime_directory)
    || !Array.isArray(value.assets)) {
    throw new Error("Humanoid controller source archive manifest is invalid");
  }
  const assets = value.assets.map((asset, index) => {
    if (!isRecord(asset)
      || !isNonEmptyString(asset.id)
      || !/^[a-z0-9][a-z0-9._-]*$/.test(asset.id)
      || !isArchiveFileName(asset.file)
      || !isSha256(asset.sha256)
      || typeof asset.bytes !== "number"
      || !Number.isSafeInteger(asset.bytes)
      || asset.bytes < 0
      || typeof asset.source_identity !== "boolean") {
      throw new Error(`Humanoid controller archive asset ${index} is invalid`);
    }
    return {
      id: asset.id,
      file: asset.file,
      sha256: asset.sha256,
      bytes: asset.bytes,
      source_identity: asset.source_identity
    };
  });
  if (new Set(assets.map(({ id }) => id)).size !== assets.length) {
    throw new Error("Humanoid controller archive asset identifiers are not unique");
  }
  return {
    protocol: CONTROLLER_SOURCE_ARCHIVE_PROTOCOL,
    source_sha256: value.source_sha256,
    entry_file: value.entry_file,
    entry_file_name: value.entry_file_name,
    entry_sha256: value.entry_sha256,
    entry_identity_sha256: value.entry_identity_sha256,
    runtime_directory: value.runtime_directory,
    assets
  };
}

async function readArchivedBytes(
  archiveDirectory: string,
  fileName: string,
  expectedSha256: string
): Promise<Uint8Array> {
  if (!isArchiveFileName(fileName)) {
    throw new Error("Humanoid controller archive contains an unsafe file name");
  }
  const bytes = await readFile(resolve(archiveDirectory, fileName));
  if (sha256(bytes) !== expectedSha256) {
    throw new Error(`Humanoid controller archive hash mismatch: ${fileName}`);
  }
  return bytes;
}

function safeRuntimeDirectory(baseDirectory: string, path: string): string {
  if (!safeRelativePath(path)) {
    throw new Error("Humanoid controller archive runtime directory is unsafe");
  }
  const root = resolve(baseDirectory);
  const directory = resolve(root, path);
  const fromRoot = relative(root, directory);
  if (!safeRelativePath(fromRoot)) {
    throw new Error("Humanoid controller archive escapes the runtime root");
  }
  return directory;
}

function safeRelativePath(path: string): boolean {
  return !isAbsolute(path)
    && !path.includes("\0")
    && !path.split(/[\\/]+/u).includes("..");
}

function isArchiveFileName(value: unknown): value is string {
  return typeof value === "string"
    && value.length > 0
    && basename(value) === value
    && !value.includes("\0");
}

function isControllerEntryFileName(value: unknown): value is string {
  return isArchiveFileName(value) && /\.(?:cjs|mjs|js)$/u.test(value);
}

function isSha256(value: unknown): value is string {
  return typeof value === "string" && SHA256_PATTERN.test(value);
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
