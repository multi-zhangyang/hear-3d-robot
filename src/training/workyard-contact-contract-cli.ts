#!/usr/bin/env node
import { resolve } from "node:path";
import { loadRuntimeCatalog } from "../config/load.js";
import {
  dryRunWorkyardContactTrainingContract,
  inspectWorkyardContactArtifacts,
  loadWorkyardContactTrainingContract
} from "./workyard-contact-contract.js";

const contractPath = resolve(
  process.argv[2] ?? "training/workyard-contact-task-v1.json"
);
const [catalog, contract] = await Promise.all([
  loadRuntimeCatalog(),
  loadWorkyardContactTrainingContract(contractPath)
]);
const artifacts = await inspectWorkyardContactArtifacts(contract, contractPath);
const scenario = catalog.materialize(contract.scenario_id, 0);
const report = dryRunWorkyardContactTrainingContract(
  contract, scenario, artifacts
);
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
if (!report.colab_training_ready) process.exitCode = 1;
