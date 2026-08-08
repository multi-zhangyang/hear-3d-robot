import assert from "node:assert/strict";
import { execFile as execFileCallback } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFile = promisify(execFileCallback);
const cliEntry = fileURLToPath(new URL("../dist/cli.js", import.meta.url));

await smokeProductionOperator();

async function smokeProductionOperator() {
  const originalWorkingDirectory = process.cwd();
  const workingDirectory = await mkdtemp(join(tmpdir(), "hear-production-smoke-"));
  const runsDir = join(workingDirectory, "runs");
  let app;

  try {
    process.chdir(workingDirectory);
    const cli = await execFile(process.execPath, [cliEntry, "scenarios"], {
      cwd: workingDirectory,
      timeout: 10_000
    });
    assert.equal(cli.stderr, "");
    assert.match(cli.stdout, /^humanoid_frontier\t/m);

    const { loadHumanoidControllerSource } = await import(
      "../dist/world/humanoid/controller-module.js"
    );
    const controllerSource = await loadHumanoidControllerSource(
      "hear/controllers/mjlab-g1-velocity",
      originalWorkingDirectory
    );
    assert.match(controllerSource.sourceSha256, /^[a-f0-9]{64}$/);
    const [firstController, secondController] = await Promise.all([
      controllerSource.controllerFactory(),
      controllerSource.controllerFactory()
    ]);
    assert.notEqual(firstController, secondController);
    assert.deepEqual(
      firstController.descriptor.learnedPolicy?.capabilities,
      ["balance", "locomotion"]
    );
    await Promise.all([firstController.dispose(), secondController.dispose()]);

    const [{ loadRuntimeCatalog }, { createOperatorServer }] = await Promise.all([
      import("../dist/config/load.js"),
      import("../dist/server/operator-server.js")
    ]);
    const catalog = await loadRuntimeCatalog();
    app = await createOperatorServer({
      server: {
        host: "127.0.0.1",
        port: 0,
        password: "",
        runsDir
      },
      catalog,
      providerError: "Provider is not configured for the production smoke test"
    });
    const origin = await app.listen({ host: "127.0.0.1", port: 0 });

    const health = await requestJson(`${origin}/api/health`);
    assert.deepEqual(health, { status: "ok" });

    const bootstrap = await requestJson(`${origin}/api/bootstrap`);
    assert.equal(bootstrap.provider?.configured, false);
    assert.equal(bootstrap.authentication_required, false);
    assert.ok(Array.isArray(bootstrap.capability_catalog));
    assert.ok(bootstrap.capability_catalog.length > 0);
    assert.ok(Array.isArray(bootstrap.scenarios));
    assert.ok(bootstrap.scenarios.length > 0);

    const root = await fetch(`${origin}/`, { signal: AbortSignal.timeout(10_000) });
    assert.equal(root.status, 200);
    assert.match(root.headers.get("content-type") ?? "", /^text\/html\b/);
    assert.match(await root.text(), /<div id="root"><\/div>/);
  } finally {
    try {
      if (app) {
        await app.close();
        assert.equal(app.server.listening, false);
      }
    } finally {
      process.chdir(originalWorkingDirectory);
      await rm(workingDirectory, { recursive: true, force: true });
    }
  }

  process.stdout.write("Production operator smoke test passed\n");
}

async function requestJson(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(10_000) });
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^application\/json\b/);
  return response.json();
}
