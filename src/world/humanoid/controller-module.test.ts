import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  HUMANOID_CONTROLLER_MODULE_ENV,
  loadConfiguredHumanoidControllerSource,
  loadHumanoidControllerSource
} from "./controller-module.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) => (
    rm(directory, { recursive: true, force: true })
  )));
});

describe("humanoid controller modules", () => {
  it("loads a relative module and creates independent controller instances", async () => {
    const directory = await temporaryDirectory();
    const entry = join(directory, "controller.mjs");
    await writeFile(entry, controllerModule("relative-policy"), "utf8");

    const source = await loadConfiguredHumanoidControllerSource({
      [HUMANOID_CONTROLLER_MODULE_ENV]: "./controller.mjs"
    }, directory);
    expect(source?.sourceSha256).toBe(
      createHash("sha256").update(await readFile(entry)).digest("hex")
    );

    const first = await source!.controllerFactory();
    const second = await source!.controllerFactory();
    expect(first).not.toBe(second);
    expect(first.descriptor.implementation).toBe("relative-policy");
    await Promise.all([first.dispose(), second.dispose()]);
  });

  it("resolves absolute paths and installed package entry points", async () => {
    const directory = await temporaryDirectory();
    const absoluteEntry = join(directory, "absolute.mjs");
    await writeFile(absoluteEntry, controllerModule("absolute-policy"), "utf8");
    const absoluteSource = await loadHumanoidControllerSource(
      absoluteEntry,
      directory
    );
    expect((await absoluteSource.controllerFactory()).descriptor.implementation)
      .toBe("absolute-policy");

    const commonJsEntry = join(directory, "commonjs.cjs");
    await writeFile(
      commonJsEntry,
      `${controllerModule("commonjs-policy").replace(
        "export function createHumanoidWholeBodyController",
        "function createHumanoidWholeBodyController"
      )}\nmodule.exports = { createHumanoidWholeBodyController };\n`,
      "utf8"
    );
    const commonJsSource = await loadHumanoidControllerSource(
      commonJsEntry,
      directory
    );
    expect((await commonJsSource.controllerFactory()).descriptor.implementation)
      .toBe("commonjs-policy");

    const packageDirectory = join(directory, "node_modules", "test-controller");
    await mkdir(packageDirectory, { recursive: true });
    await writeFile(join(packageDirectory, "package.json"), JSON.stringify({
      name: "test-controller",
      type: "module",
      exports: "./index.mjs"
    }), "utf8");
    await writeFile(
      join(packageDirectory, "index.mjs"),
      controllerModule("package-policy"),
      "utf8"
    );
    const packageSource = await loadHumanoidControllerSource(
      "test-controller",
      directory
    );
    expect((await packageSource.controllerFactory()).descriptor.implementation)
      .toBe("package-policy");
  });

  it("rejects missing exports and invalid controller contracts", async () => {
    const directory = await temporaryDirectory();
    const missing = join(directory, "missing.mjs");
    await writeFile(missing, "export const unrelated = true;\n", "utf8");
    await expect(loadHumanoidControllerSource(missing, directory)).rejects.toThrow(
      /createHumanoidWholeBodyController/
    );

    const invalid = join(directory, "invalid.mjs");
    await writeFile(invalid, [
      "export function createHumanoidWholeBodyController() {",
      "  return { descriptor: { protocol: 'humanoid-controller-v1' } };",
      "}"
    ].join("\n"), "utf8");
    const source = await loadHumanoidControllerSource(invalid, directory);
    await expect(source.controllerFactory()).rejects.toThrow(/missing method: reset/);
  });

  it("rejects a module that shares one stateful controller instance", async () => {
    const directory = await temporaryDirectory();
    const entry = join(directory, "shared.mjs");
    await writeFile(entry, `${controllerModule("shared-policy", true)}\n`, "utf8");
    const source = await loadHumanoidControllerSource(entry, directory);

    await source.controllerFactory();
    await expect(source.controllerFactory()).rejects.toThrow(/previously used stateful instance/);
  });

  it("changes the persisted identity when module entry content changes", async () => {
    const directory = await temporaryDirectory();
    const entry = join(directory, "changing.mjs");
    await writeFile(entry, controllerModule("policy-v1"), "utf8");
    const first = await loadHumanoidControllerSource(entry, directory);
    await writeFile(entry, controllerModule("policy-v2"), "utf8");
    const second = await loadHumanoidControllerSource(entry, directory);

    expect(first.sourceSha256).not.toBe(second.sourceSha256);
  });

  it("leaves the built-in controller selected when no module is configured", async () => {
    await expect(loadConfiguredHumanoidControllerSource({})).resolves.toBeUndefined();
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "hear-controller-module-"));
  temporaryDirectories.push(directory);
  return directory;
}

function controllerModule(implementation: string, shared = false): string {
  const factoryBody = shared
    ? "return sharedController;"
    : "return createController(context);";
  return `
function createController(context) {
  return {
    descriptor: {
      protocol: "humanoid-controller-v1",
      implementation: ${JSON.stringify(implementation)},
      actuation: "joint_position_pd",
      controlStepSeconds: 0.02,
      physicsStepSeconds: 0.002
    },
    reset() {},
    async infer() {
      return {
        kind: "joint_position_pd",
        positions: new Float64Array(29),
        stiffness: new Float64Array(29),
        damping: new Float64Array(29)
      };
    },
    advanceHistory() {},
    captureState() {
      return {
        protocol: "humanoid-controller-state-v1",
        version: 1,
        implementation: ${JSON.stringify(implementation)},
        payload: { source_sha256: context.sourceSha256 }
      };
    },
    restoreState() {},
    async dispose() {}
  };
}
const sharedController = createController({ sourceSha256: "shared" });
export function createHumanoidWholeBodyController(context) {
  if (context.protocol !== "hear-humanoid-controller-module-v1") {
    throw new Error("Unexpected module protocol");
  }
  ${factoryBody}
}
`;
}
