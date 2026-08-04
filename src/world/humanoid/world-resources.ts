import type { Scenario, Vec3 } from "../../domain/schema.js";
import {
  HUMANOID_NAVIGATION_PROFILE,
  humanoidEnvironment
} from "./environment.js";
import { HumanoidNavigationPlanner } from "./navigation-planner.js";
import {
  humanoidPhysicalRegion,
  type HumanoidPhysicalRegion
} from "./physical-region.js";
import { HumanoidRolloutSimulationPool } from "./rollout-simulation-pool.js";
import { HumanoidSimulation } from "./simulation.js";

export interface HumanoidWorldResources {
  simulation: HumanoidSimulation;
  rolloutPool: HumanoidRolloutSimulationPool;
  navigation: HumanoidNavigationPlanner;
  physicalRegion: HumanoidPhysicalRegion;
}

export interface HumanoidPhysicsResources {
  simulation: HumanoidSimulation;
  rolloutPool: HumanoidRolloutSimulationPool;
  physicalRegion: HumanoidPhysicalRegion;
}

export async function createHumanoidWorldResources(
  scenario: Scenario,
  anchors: readonly Pick<Vec3, "x" | "z">[] = [{
    x: scenario.robot.x,
    z: scenario.robot.z
  }]
): Promise<HumanoidWorldResources> {
  const navigation = new HumanoidNavigationPlanner(
    scenario,
    HUMANOID_NAVIGATION_PROFILE
  );
  try {
    const physics = await createHumanoidPhysicsResources(scenario, anchors);
    return { ...physics, navigation };
  } catch (error) {
    await navigation.dispose();
    throw error;
  }
}

export async function createHumanoidPhysicsResources(
  scenario: Scenario,
  anchors: readonly Pick<Vec3, "x" | "z">[]
): Promise<HumanoidPhysicsResources> {
  const physicalRegion = humanoidPhysicalRegion(scenario, anchors);
  const environment = humanoidEnvironment(scenario, physicalRegion);
  let simulation: HumanoidSimulation | undefined;
  let rolloutPool: HumanoidRolloutSimulationPool | undefined;
  try {
    [simulation, rolloutPool] = await Promise.all([
      HumanoidSimulation.create(environment),
      HumanoidRolloutSimulationPool.create(environment)
    ]);
    return { simulation, rolloutPool, physicalRegion };
  } catch (error) {
    await Promise.allSettled([
      ...(simulation ? [simulation.dispose()] : []),
      ...(rolloutPool ? [rolloutPool.dispose()] : [])
    ]);
    throw error;
  }
}

export async function disposeHumanoidPhysicsResources(
  resources: HumanoidPhysicsResources
): Promise<void> {
  await Promise.all([
    resources.simulation.dispose(),
    resources.rolloutPool.dispose()
  ]);
}

export async function disposeHumanoidWorldResources(
  resources: HumanoidWorldResources
): Promise<void> {
  await Promise.all([
    resources.navigation.dispose(),
    disposeHumanoidPhysicsResources(resources)
  ]);
}
