#!/usr/bin/env node
import { resolve } from "node:path";
import { loadRuntimeCatalog } from "../config/load.js";
import {
  dryRunWorkyardResidualTrainingContract,
  inspectWorkyardResidualTeacherArtifacts,
  loadWorkyardResidualTrainingContract
} from "./workyard-residual-contract.js";

const contractPath = resolve(process.argv[2] ?? "training/workyard-task-v4.json");
const teacherRoot = resolve(
  process.argv[3] ?? "artifacts/training/g1-residual-teacher"
);
const [catalog, contract, teacher] = await Promise.all([
  loadRuntimeCatalog(),
  loadWorkyardResidualTrainingContract(contractPath),
  inspectWorkyardResidualTeacherArtifacts(teacherRoot)
]);
const scenario = catalog.materialize(contract.scenario_id, 0);
const report = dryRunWorkyardResidualTrainingContract(contract, scenario, teacher);
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
if (!report.colab_smoke_ready) process.exitCode = 1;
