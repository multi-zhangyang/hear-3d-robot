import type { FastifyInstance } from "fastify";
import {
  loadProviderConfig,
  loadRuntimeCatalog,
  loadServerConfig,
  type ProviderConfig
} from "../../src/config/load.js";
import { errorMessage } from "../../src/runtime/error-message.js";
import { createOperatorServer } from "../../src/server/operator-server.js";
import { prepareE2ERuns } from "./e2e-runs.js";

await prepareE2ERuns();

const [catalog, server] = await Promise.all([
  loadRuntimeCatalog(),
  Promise.resolve(loadServerConfig())
]);
let provider: ProviderConfig | undefined;
let providerError: string | undefined;
try {
  provider = loadProviderConfig();
} catch (error) {
  providerError = errorMessage(error);
}

const app = await createOperatorServer({
  server,
  catalog,
  ...(provider ? { provider } : {}),
  ...(providerError ? { providerError } : {})
});
installShutdownHandlers(app);
try {
  await app.listen({ host: server.host, port: server.port });
} catch (error) {
  await app.close();
  throw error;
}

function installShutdownHandlers(operator: FastifyInstance): void {
  let closing: Promise<void> | undefined;
  const close = (): void => {
    closing ??= operator.close().catch((error) => {
      process.stderr.write(`${errorMessage(error)}\n`);
      process.exitCode = 1;
    });
  };
  process.on("SIGINT", close);
  process.on("SIGTERM", close);
  operator.addHook("onClose", async () => {
    process.off("SIGINT", close);
    process.off("SIGTERM", close);
  });
}
