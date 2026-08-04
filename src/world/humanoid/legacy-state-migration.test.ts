import { describe, expect, it } from "vitest";
import { HUMANOID_JOINT_NAMES } from "./model.js";
import {
  looksLikeLegacyG129DoFState,
  migrateLegacyG129DoFState
} from "./legacy-state-migration.js";
import type {
  MujocoActuatedJointBinding,
  MujocoJointBinding
} from "./mujoco-joints.js";

describe("legacy G1 physical state migration", () => {
  it("maps shuffled body and object bindings by their stable names", () => {
    const bodyBindings = [...HUMANOID_JOINT_NAMES].reverse().map((name) => {
      const index = HUMANOID_JOINT_NAMES.indexOf(name);
      return {
        name,
        jointId: 100 + index,
        positionAddress: 7 + index,
        velocityAddress: 6 + index,
        actuatorId: index
      } satisfies MujocoActuatedJointBinding;
    });
    const objectBindings: MujocoJointBinding[] = [
      {
        name: "world-object-joint-1",
        jointId: 201,
        positionAddress: 57,
        velocityAddress: 55
      },
      {
        name: "world-object-joint-0",
        jointId: 200,
        positionAddress: 50,
        velocityAddress: 49
      }
    ];
    const source = {
      positions: Float64Array.from({ length: 50 }, (_, index) => 1000 + index),
      velocities: Float64Array.from({ length: 47 }, (_, index) => 2000 + index),
      controls: Float64Array.from({ length: 29 }, (_, index) => 3000 + index),
      activations: new Float64Array(0),
      accelerationWarmstart: Float64Array.from(
        { length: 47 },
        (_, index) => 4000 + index
      )
    };
    const target = {
      positions: new Float64Array(64).fill(-101),
      velocities: new Float64Array(61).fill(-102),
      controls: new Float64Array(43).fill(-103),
      activations: new Float64Array(0),
      accelerationWarmstart: new Float64Array(61).fill(-104)
    };

    expect(looksLikeLegacyG129DoFState({ source, target })).toBe(true);
    const migrated = migrateLegacyG129DoFState({
      source,
      target,
      bodyBindings,
      objectBindings
    });

    expect([...migrated.positions.slice(0, 7)]).toEqual([...source.positions.slice(0, 7)]);
    expect([...migrated.velocities.slice(0, 6)]).toEqual([...source.velocities.slice(0, 6)]);
    HUMANOID_JOINT_NAMES.forEach((_name, index) => {
      expect(migrated.positions[7 + index]).toBe(source.positions[7 + index]);
      expect(migrated.velocities[6 + index]).toBe(source.velocities[6 + index]);
      expect(migrated.controls[index]).toBe(source.controls[index]);
      expect(migrated.accelerationWarmstart[6 + index]).toBe(
        source.accelerationWarmstart[6 + index]
      );
    });
    expect([...migrated.positions.slice(36, 50)]).toEqual(new Array(14).fill(-101));
    expect([...migrated.velocities.slice(35, 49)]).toEqual(new Array(14).fill(-102));
    expect([...migrated.controls.slice(29)]).toEqual(new Array(14).fill(-103));
    expect([...migrated.accelerationWarmstart.slice(35, 49)])
      .toEqual(new Array(14).fill(-104));
    expect([...migrated.positions.slice(50, 57)]).toEqual(
      [...source.positions.slice(36, 43)]
    );
    expect([...migrated.positions.slice(57, 64)]).toEqual(
      [...source.positions.slice(43, 50)]
    );
    expect([...migrated.velocities.slice(49, 55)]).toEqual(
      [...source.velocities.slice(35, 41)]
    );
    expect([...migrated.velocities.slice(55, 61)]).toEqual(
      [...source.velocities.slice(41, 47)]
    );
  });

  it("rejects object bindings that cannot be tied to a legacy object index", () => {
    const source = {
      positions: new Float64Array(43),
      velocities: new Float64Array(41),
      controls: new Float64Array(29),
      activations: new Float64Array(0),
      accelerationWarmstart: new Float64Array(41)
    };
    const target = {
      positions: new Float64Array(57),
      velocities: new Float64Array(55),
      controls: new Float64Array(43),
      activations: new Float64Array(0),
      accelerationWarmstart: new Float64Array(55)
    };
    const bodyBindings = HUMANOID_JOINT_NAMES.map((name, index) => ({
      name,
      jointId: index,
      positionAddress: 7 + index,
      velocityAddress: 6 + index,
      actuatorId: index
    }));

    expect(() => migrateLegacyG129DoFState({
      source,
      target,
      bodyBindings,
      objectBindings: [{
        name: "anonymous-free-joint",
        jointId: 99,
        positionAddress: 50,
        velocityAddress: 49
      }]
    })).toThrow(/invalid named layout/);
  });
});
