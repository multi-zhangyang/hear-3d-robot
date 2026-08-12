#!/usr/bin/env node
import { resolve } from "node:path";
import { loadRuntimeCatalog } from "../config/load.js";
import {
  dryRunWorkyardTrainingContract,
  loadWorkyardTrainingContract
} from "./workyard-contract.js";

const contractPath = resolve(process.argv[2] ?? "training/workyard-task-v2.json");
const [catalog, contract] = await Promise.all([
  loadRuntimeCatalog(),
  loadWorkyardTrainingContract(contractPath)
]);
const scenario = catalog.materialize(contract.scenario_id, 0);
const report = dryRunWorkyardTrainingContract(contract, scenario);
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
