import { Mutex } from "async-mutex";
import type { Scenario, Vec3 } from "../../domain/schema.js";
import { navigationCorridorBuildStages } from "../navigation-corridor.js";
import {
  NavigationMesh,
  NavigationPlanningError,
  type NavigationAgentProfile,
  type NavigationObstacle,
  type NavigationPlan
} from "../navigation.js";

const MAXIMUM_CACHED_NAVIGATION_SCOPES = 2;

interface CachedNavigationScope {
  mesh: NavigationMesh;
  lastUsed: number;
}

export interface HumanoidNavigationPlannerState {
  cachedScopeCount: number;
  buildCount: number;
  lastExpansionTiles: number | null;
  lastSelectedTileCount: number | null;
  lastTotalTileCount: number | null;
}

/**
 * Builds Recast only around the requested route and widens that corridor when
 * a valid local mesh cannot produce a path. Planning is serialized because a
 * Recast tile cache is mutable while dynamic obstacles are synchronized.
 */
export class HumanoidNavigationPlanner {
  readonly #scenario: Scenario;
  readonly #profile: NavigationAgentProfile;
  readonly #mutex = new Mutex();
  readonly #cache = new Map<string, CachedNavigationScope>();
  #sequence = 0;
  #buildCount = 0;
  #disposed = false;
  #lastStage: {
    expansionTiles: number | null;
    selectedTileCount: number;
    totalTileCount: number;
  } | null = null;

  constructor(scenario: Scenario, profile: NavigationAgentProfile) {
    this.#scenario = structuredClone(scenario);
    this.#profile = structuredClone(profile);
  }

  plan(
    start: Vec3,
    target: Vec3,
    obstacles: readonly NavigationObstacle[]
  ): Promise<NavigationPlan> {
    return this.#mutex.runExclusive(async () => {
      this.#assertAvailable();
      const stages = navigationCorridorBuildStages(this.#scenario, start, target);
      let lastPlanningFailure: NavigationPlanningError | undefined;
      for (const stage of stages) {
        const mesh = await this.#mesh(stage.key, stage.scope);
        try {
          const plan = mesh.plan(start, target, obstacles);
          this.#lastStage = {
            expansionTiles: stage.expansionTiles,
            selectedTileCount: stage.selectedTileCount,
            totalTileCount: stage.totalTileCount
          };
          return plan;
        } catch (error) {
          if (!(error instanceof NavigationPlanningError)) throw error;
          lastPlanningFailure = error;
        }
      }
      throw lastPlanningFailure ?? new NavigationPlanningError(
        "path_not_found",
        "No navigation corridor could be built for the requested route"
      );
    });
  }

  state(): HumanoidNavigationPlannerState {
    return {
      cachedScopeCount: this.#cache.size,
      buildCount: this.#buildCount,
      lastExpansionTiles: this.#lastStage?.expansionTiles ?? null,
      lastSelectedTileCount: this.#lastStage?.selectedTileCount ?? null,
      lastTotalTileCount: this.#lastStage?.totalTileCount ?? null
    };
  }

  async dispose(): Promise<void> {
    await this.#mutex.runExclusive(() => {
      if (this.#disposed) return;
      this.#disposed = true;
      for (const cached of this.#cache.values()) cached.mesh.dispose();
      this.#cache.clear();
    });
  }

  async #mesh(
    key: string,
    scope: Parameters<typeof NavigationMesh.create>[1]
  ): Promise<NavigationMesh> {
    const cached = this.#cache.get(key);
    if (cached) {
      cached.lastUsed = ++this.#sequence;
      return cached.mesh;
    }
    const mesh = await NavigationMesh.create(this.#scenario, scope, this.#profile);
    if (this.#disposed) {
      mesh.dispose();
      throw new Error("Humanoid navigation planner is disposed");
    }
    this.#cache.set(key, { mesh, lastUsed: ++this.#sequence });
    this.#buildCount += 1;
    this.#evictLeastRecentlyUsed(key);
    return mesh;
  }

  #evictLeastRecentlyUsed(retainedKey: string): void {
    while (this.#cache.size > MAXIMUM_CACHED_NAVIGATION_SCOPES) {
      const candidate = [...this.#cache.entries()]
        .filter(([key]) => key !== retainedKey)
        .sort((left, right) => left[1].lastUsed - right[1].lastUsed)[0];
      if (!candidate) return;
      candidate[1].mesh.dispose();
      this.#cache.delete(candidate[0]);
    }
  }

  #assertAvailable(): void {
    if (this.#disposed) throw new Error("Humanoid navigation planner is disposed");
  }
}
