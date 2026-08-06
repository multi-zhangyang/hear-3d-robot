import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import type { G1HandCoordination } from "./hand-coordination.js";
import { contactAwareG1GraspTargetsForOption } from "./contact-aware-grasp-servo.js";
import { HumanoidGraspRegistry } from "./grasp-registry.js";
import { applyHumanoidMotionArtifactFrame } from "./motion-frame-application.js";
import { prepareHumanoidMotion, type HumanoidMotionPlan } from "./motion-plan.js";
import { G1_HAND_CONTACT_SURFACE_NAMES } from "./morphology.js";
import { neutralHumanoidReference } from "./reference.js";
import { HumanoidSimulation } from "./simulation.js";
import { HumanoidWorld } from "./world.js";
import { assessHumanoidObjectReleased } from "./object-release.js";
import { assessHumanoidObjectSettledOnSupport } from "./object-settled-support.js";
import {
  inverseQuaternion,
  multiplyQuaternion,
  normalizeQuaternion,
  rotateVector,
  subtract
} from "../geometry.js";

const OBJECT_ID = "grasp-rod";
const OBJECT_CENTER = { x: 1.693, y: 0.695, z: 1.735 };
const OBJECT_SIZE = { x: 0.03, y: 0.27, z: 0.03 };
const SUPPORT_SIZE = { x: 0.05, y: 0.01, z: 0.05 };
const SUPPORT_TOP = OBJECT_CENTER.y - OBJECT_SIZE.y / 2;

const SCENARIO = ScenarioSchema.parse({
  title: "Physical grasp transport field",
  seed: 19,
  bounds: { width: 6, depth: 6 },
  visibility_radius: 4,
  robot: { x: 1.5, z: 1.5, yaw: 0 },
  obstacles: [{
    id: "grasp-support",
    center: {
      x: OBJECT_CENTER.x,
      y: SUPPORT_TOP - 0.005,
      z: OBJECT_CENTER.z
    },
    size: SUPPORT_SIZE
  }],
  objects: [{
    id: OBJECT_ID,
    kind: "rod",
    color: "#b77a42",
    position: OBJECT_CENTER,
    size: OBJECT_SIZE,
    portable: true
  }],
  zones: [{
    id: "drop-zone",
    color: "#62c98d",
    center: { x: 1.5, y: -0.025, z: 2.6 },
    size: { x: 1.4, y: 0.05, z: 1.6 }
  }],
  default_goal: {
    summary: "抓取并抬升物体",
    predicates: [{
      type: "object_grasped",
      object_id: OBJECT_ID,
      hand: "left"
    }]
  }
});

const OPEN_HANDS: G1HandCoordination = {
  left: {
    thumb_opposition: 0,
    thumb_curl: 0,
    index_curl: 0,
    middle_curl: 0
  },
  right: {
    thumb_opposition: 0,
    thumb_curl: 0,
    index_curl: 0,
    middle_curl: 0
  }
};

const CLOSED_LEFT_HAND: G1HandCoordination = {
  left: {
    thumb_opposition: 0.7,
    thumb_curl: 0.8,
    index_curl: 0.8,
    middle_curl: 0.8
  },
  right: OPEN_HANDS.right
};

describe("humanoid physical grasp transport", () => {
  it("lifts a supported object through opposed G1 hand contacts", async () => {
    const simulation = await HumanoidSimulation.create({
      spawn: {
        position: { x: 1.5, y: 0, z: 1.5 },
        yaw: 0
      },
      solids: [{
        id: "grasp-support",
        center: {
          x: OBJECT_CENTER.x,
          y: SUPPORT_TOP - 0.005,
          z: OBJECT_CENTER.z
        },
        size: SUPPORT_SIZE
      }],
      objects: [{
        id: OBJECT_ID,
        center: OBJECT_CENTER,
        size: OBJECT_SIZE,
        mass: 0.25
      }]
    });
    try {
      const reference = neutralHumanoidReference();
      const registry = new HumanoidGraspRegistry({
        portableObjectIds: [OBJECT_ID]
      });
      let frame = 0;
      let snapshot = simulation.snapshot();
      registry.observe(frame, snapshot);
      for (let index = 0; index < 100; index += 1) {
        snapshot = await simulation.step(reference);
        registry.observe(++frame, snapshot);
      }

      const initialObjectHeight = snapshot.objects[OBJECT_ID]!.position.y;
      const wrist = rotateVector(
        inverseQuaternion(snapshot.links.pelvis.rotation),
        subtract(
          snapshot.links.left_wrist_yaw_link.position,
          snapshot.links.pelvis.position
        )
      );
      const liftedWristPosition = {
        x: wrist.x + 0.1,
        y: wrist.y + 0.052,
        z: wrist.z + 0.015
      };
      const plan: HumanoidMotionPlan = {
        id: "physical-left-grasp-lift",
        intent: "闭合左手并保持真实对向接触后抬升物体",
        duration_seconds: 2.65,
        contact_constraints: G1_HAND_CONTACT_SURFACE_NAMES
          .filter((surface) => surface.startsWith("left_"))
          .map((hand_surface) => ({
            hand_surface,
            object_id: OBJECT_ID,
            required: false
          })),
        keyframes: [{
          at_seconds: 0,
          hand_coordination: OPEN_HANDS
        }, {
          at_seconds: 1.4,
          hand_coordination: CLOSED_LEFT_HAND
        }, {
          at_seconds: 1.8,
          hand_coordination: CLOSED_LEFT_HAND
        }, {
          at_seconds: 2.3,
          hand_coordination: CLOSED_LEFT_HAND,
          left_hand: {
            frame: "pelvis",
            position: liftedWristPosition,
            tolerance_m: 0.075
          }
        }, {
          at_seconds: 2.65,
          hand_coordination: CLOSED_LEFT_HAND,
          left_hand: {
            frame: "pelvis",
            position: liftedWristPosition,
            tolerance_m: 0.075
          }
        }]
      };
      const option = {
        option_id: "physical-left-grasp-lift",
        predicates: [{
          type: "grasp_verified" as const,
          object_id: OBJECT_ID,
          hand: "left" as const,
          grasp_contract_sha256: registry.contractSha256
        }],
        stable_steps: 1
      };

      const prepared = await prepareHumanoidMotion(
        simulation,
        plan,
        reference,
        {
          contactObjectIds: new Set([OBJECT_ID]),
          graspRegistry: registry,
          worldFrame: frame,
          motionOption: {
            scenario: SCENARIO,
            contract: option
          }
        }
      );
      expect(
        prepared.validation.feasible,
        JSON.stringify(prepared.validation.failures)
      ).toBe(true);
      expect(prepared.artifact).not.toBeNull();
      expect(prepared.rollout).not.toBeNull();
      expect(prepared.optionCertificate).not.toBeNull();

      const graspTargets = contactAwareG1GraspTargetsForOption({
        option,
        graspContract: registry.contract
      });

      let verifiedFrames = 0;
      let maximumLift = 0;
      let maximumStableFrames = 0;
      let maximumLiftedHoldFrames = 0;
      let mostStableState: unknown = null;
      const reasons = new Map<string, number>();
      for (const artifactFrame of prepared.artifact!.frames.slice(
        0,
        prepared.optionCertificate!.validated_frame_limit
      )) {
        snapshot = (
          await applyHumanoidMotionArtifactFrame(simulation, artifactFrame, {
            graspTargets
          })
        ).snapshot;
        const assessments = registry.observe(++frame, snapshot);
        const left = assessments.find((assessment) => assessment.hand === "left");
        if (left) {
          if (left.grasp_verified) verifiedFrames += 1;
          if (left.evidence.relative_pose.stable_frames > maximumStableFrames) {
            maximumStableFrames = left.evidence.relative_pose.stable_frames;
            mostStableState = {
              atSeconds: artifactFrame.atSeconds,
              assessment: left,
              object: snapshot.objects[OBJECT_ID],
              wrist: snapshot.links.left_wrist_yaw_link,
              pelvis: snapshot.links.pelvis
            };
          }
          maximumLiftedHoldFrames = Math.max(
            maximumLiftedHoldFrames,
            left.evidence.lifted_hold_frames
          );
          reasons.set(left.reason, (reasons.get(left.reason) ?? 0) + 1);
        }
        maximumLift = Math.max(
          maximumLift,
          snapshot.objects[OBJECT_ID]!.position.y - initialObjectHeight
        );
      }

      const finalAssessment = registry.assessmentsForFrame(frame).find(
        (assessment) => assessment.hand === "left"
      );
      expect(maximumLift).toBeGreaterThanOrEqual(
        registry.contract.minimum_lift_m
      );
      const diagnostics = JSON.stringify({
        maximumLift,
        maximumStableFrames,
        maximumLiftedHoldFrames,
        mostStableState,
        reasons: Object.fromEntries(reasons),
        object: snapshot.objects[OBJECT_ID],
        wrist,
        finalAssessment,
        contacts: snapshot.contacts.filter((contact) => (
          contact.firstObject === OBJECT_ID || contact.secondObject === OBJECT_ID
        ))
      });
      expect(maximumLiftedHoldFrames, diagnostics).toBeGreaterThanOrEqual(
        registry.contract.minimum_lifted_hold_frames
      );
      expect(verifiedFrames, diagnostics).toBeGreaterThanOrEqual(1);
      expect(finalAssessment).toMatchObject({
        object_id: OBJECT_ID,
        hand: "left",
        grasp_verified: true,
        reason: "grasp_verified"
      });
      expect(
        finalAssessment!.evidence.contact.distinct_force_qualified_links.length
      )
        .toBeGreaterThanOrEqual(registry.contract.minimum_distinct_contact_links);
    } finally {
      await simulation.dispose();
    }
  }, 60_000);

  it("executes the certified grasp prefix and publishes a carried-object binding", async () => {
    const world = await HumanoidWorld.create(SCENARIO);
    try {
      const observed = world.observe();
      expect(observed.objectTokens.some((object) => object.id === OBJECT_ID)).toBe(true);
      const before = world.snapshot();
      const wrist = rotateVector(
        inverseQuaternion(before.robot.links.pelvis.rotation),
        subtract(
          before.robot.links.left_wrist_yaw_link.position,
          before.robot.links.pelvis.position
        )
      );
      const surfaces = G1_HAND_CONTACT_SURFACE_NAMES
        .filter((surface) => surface.startsWith("left_"))
        .map((hand_surface) => ({
          hand_surface,
          object_id: OBJECT_ID,
          required: false as const
        }));
      const plan = (input: {
        id: string;
        closeAt: number;
        settleUntil: number;
        liftAt: number;
        endAt: number;
        liftX: number;
        liftY: number;
        liftZ: number;
      }): HumanoidMotionPlan => ({
        id: input.id,
        intent: "以左手真实对向接触稳定抓取并抬升物体",
        duration_seconds: input.endAt,
        contact_constraints: surfaces,
        keyframes: [{
          at_seconds: 0,
          hand_coordination: OPEN_HANDS
        }, {
          at_seconds: input.closeAt,
          hand_coordination: CLOSED_LEFT_HAND
        }, {
          at_seconds: input.settleUntil,
          hand_coordination: CLOSED_LEFT_HAND
        }, {
          at_seconds: input.liftAt,
          hand_coordination: CLOSED_LEFT_HAND,
          left_hand: {
            frame: "pelvis",
            position: {
              ...wrist,
              x: wrist.x + input.liftX,
              y: wrist.y + input.liftY,
              z: wrist.z + input.liftZ
            },
            tolerance_m: 0.075
          }
        }, {
          at_seconds: input.endAt,
          hand_coordination: CLOSED_LEFT_HAND,
          left_hand: {
            frame: "pelvis",
            position: {
              ...wrist,
              x: wrist.x + input.liftX,
              y: wrist.y + input.liftY,
              z: wrist.z + input.liftZ
            },
            tolerance_m: 0.075
          }
        }]
      });
      const planned = await world.planWholeBodyMotionCandidates({
        objective: "抓取并抬升支撑面上的物体",
        termination: {
          option_id: "world-left-grasp-lift",
          predicates: [{
            type: "grasp_verified",
            object_id: OBJECT_ID,
            hand: "left",
            grasp_contract_sha256: before.grasp.contractSha256
          }],
          stable_steps: 1
        },
        candidates: [
          plan({
            id: "world-left-grasp-primary",
            closeAt: 1.3,
            settleUntil: 1.7,
            liftAt: 2.2,
            endAt: 2.6,
            liftX: 0.09,
            liftY: 0.05,
            liftZ: 0.02
          }),
          plan({
            id: "world-left-grasp-secondary",
            closeAt: 1.4,
            settleUntil: 1.8,
            liftAt: 2.3,
            endAt: 2.65,
            liftX: 0.1,
            liftY: 0.052,
            liftZ: 0.015
          })
        ]
      });
      expect(planned.accepted, JSON.stringify(planned.candidates)).toBe(true);
      expect(planned.option?.certificate.validated_frame_limit).toBe(
        planned.option?.certificate.predicted_termination_frame
      );
      expect(planned.option!.certificate.validated_frame_limit).toBeLessThan(
        planned.motion!.frame_count
      );

      const executed = await world.executeWholeBodyMotion(planned.planId);
      expect(executed).toMatchObject({
        accepted: true,
        code: "motion_option_succeeded",
        detail: {
          option: {
            status: "succeeded",
            termination_reason: "physical_success",
            actual_termination_frame:
              planned.option!.certificate.predicted_termination_frame
          }
        }
      });
      expect(executed.frames).toBe(
        planned.option!.certificate.validated_frame_limit
      );
      expect(executed.finalSnapshot.grasp.assessments).toContainEqual(
        expect.objectContaining({
          object_id: OBJECT_ID,
          hand: "left",
          grasp_verified: true,
          reason: "grasp_verified"
        })
      );

      const acquiredPelvis = executed.finalSnapshot.robot.links.pelvis;
      const acquiredWrist = rotateVector(
        inverseQuaternion(acquiredPelvis.rotation),
        subtract(
          executed.finalSnapshot.robot.links.left_wrist_yaw_link.position,
          acquiredPelvis.position
        )
      );
      const acquiredWristOrientation = normalizeQuaternion(multiplyQuaternion(
        inverseQuaternion(acquiredPelvis.rotation),
        executed.finalSnapshot.robot.links.left_wrist_yaw_link.rotation
      ));
      const acquiredRoot = executed.finalSnapshot.robot.rootPosition;
      const carriedReferencePose = world.checkpoint().graspRegistry.tracker.tracks.find(
        (track) => track.object_id === OBJECT_ID && track.hand === "left"
      )?.attempt?.reference_relative_pose;
      if (!carriedReferencePose) throw new Error("Missing certified carried pose");
      const carryClearanceTarget = {
        x: acquiredRoot.x - 0.24,
        y: acquiredRoot.y,
        z: acquiredRoot.z - 0.24
      };
      const safeRetreatPlan = (input: {
        id: string;
        forward: number;
        lateral: number;
      }): HumanoidMotionPlan => ({
        id: input.id,
        intent: "保持真实抓取并安全退离支撑物",
        duration_seconds: 3,
        contact_constraints: surfaces,
        keyframes: [{
          at_seconds: 0,
          root_velocity: {
            forward_mps: 0,
            lateral_mps: 0
          },
          hand_coordination: CLOSED_LEFT_HAND,
          left_hand: {
            frame: "pelvis",
            position: acquiredWrist,
            tolerance_m: 0.04,
            orientation: acquiredWristOrientation,
            orientation_tolerance_rad: 0.15
          }
        }, {
          at_seconds: 0.6,
          root_velocity: {
            forward_mps: input.forward,
            lateral_mps: input.lateral
          },
          hand_coordination: CLOSED_LEFT_HAND
        }, {
          at_seconds: 2.4,
          root_velocity: {
            forward_mps: input.forward,
            lateral_mps: input.lateral
          },
          hand_coordination: CLOSED_LEFT_HAND
        }, {
          at_seconds: 3,
          root_velocity: { forward_mps: 0, lateral_mps: 0 },
          hand_coordination: CLOSED_LEFT_HAND
        }]
      });
      const carryPose = await world.planWholeBodyMotionCandidates({
        objective: "从真实抓取终态选择安全退离动作",
        termination: {
          option_id: "world-left-carry-pose",
          predicates: [{
            type: "grasp_verified",
            object_id: OBJECT_ID,
            hand: "left",
            grasp_contract_sha256: before.grasp.contractSha256
          }, {
            type: "root_near_point",
            target: carryClearanceTarget,
            tolerance_m: 0.1
          }],
          stable_steps: 4,
          phases: {
            precondition: {
              condition: { op: "predicate", predicate_index: 0 },
              stable_steps: 1
            },
            during: {
              condition: { op: "predicate", predicate_index: 0 }
            },
            terminal: {
              condition: {
                op: "all",
                conditions: [{ op: "predicate", predicate_index: 0 }, {
                  op: "predicate",
                  predicate_index: 1
                }]
              }
            }
          }
        },
        candidates: [
          safeRetreatPlan({ id: "safe-retreat-backward", forward: -0.3, lateral: 0 }),
          safeRetreatPlan({ id: "safe-retreat-diagonal", forward: -0.35, lateral: -0.35 }),
          safeRetreatPlan({ id: "safe-retreat-fast", forward: -0.45, lateral: -0.25 })
        ]
      });
      expect(carryPose.accepted, JSON.stringify(carryPose.candidates.map((candidate) => ({
        rank: candidate.rank,
        planId: candidate.planId,
        failures: candidate.validation.failures,
        simulatedSteps: candidate.validation.evidence.simulatedSteps,
        travelledDistance: candidate.validation.evidence.travelledDistance,
        rootPosition: candidate.validation.finalSnapshot.rootPosition,
        leftHipPosition:
          candidate.validation.finalSnapshot.links.left_hip_roll_link.position,
        relativePoseError: relativePoseError(
          candidate.validation.finalSnapshot,
          carriedReferencePose
        ),
        objectContacts: candidate.validation.finalSnapshot.contacts.filter(
          (contact) => contact.firstObject === OBJECT_ID
            || contact.secondObject === OBJECT_ID
        )
      })))).toBe(true);
      const posed = await world.executeWholeBodyMotion(carryPose.planId);
      expect(posed).toMatchObject({
        accepted: true,
        code: "motion_option_succeeded",
        detail: {
          carry: {
            binding_set: {
              source_frame: executed.finalSnapshot.frame,
              bindings: [{ object_id: OBJECT_ID, hand: "left" }]
            },
            continuation: { continued: true },
            unauthorized_contacts: []
          }
        }
      });

      const root = posed.finalSnapshot.robot.rootPosition;
      const objectBeforeCarry = posed.finalSnapshot.robot.objects[OBJECT_ID]!.position;
      const navigation = await world.planNavigation({
        x: root.x,
        y: 0,
        z: root.z + 0.6
      });
      expect(navigation.accepted, JSON.stringify({
        reason: navigation.reason,
        carriedReferencePose,
        sourceFrame: posed.finalSnapshot.frame,
        sourceAssessment: posed.finalSnapshot.grasp.assessments.find(
          (assessment) => assessment.object_id === OBJECT_ID
            && assessment.hand === "left"
        ),
        handJoints: posed.finalSnapshot.robot.hands.joints,
        objectContacts: posed.finalSnapshot.robot.contacts.filter((contact) => (
          contact.firstObject === OBJECT_ID || contact.secondObject === OBJECT_ID
        ))
      })).toBe(true);
      expect(navigation.carry.bindings).toEqual([{
        object_id: OBJECT_ID,
        hand: "left"
      }]);

      let invalidCarryFrame: number | null = null;
      const carryTrace: unknown[] = [];
      const carried = await world.executeNavigation(
        navigation.planId,
        (snapshot) => {
          const assessment = snapshot.grasp.assessments.find((entry) => (
            entry.object_id === OBJECT_ID && entry.hand === "left"
          ));
          if (!assessment?.grasp_verified) invalidCarryFrame ??= snapshot.frame;
          if ((snapshot.frame - posed.finalSnapshot.frame) % 10 === 0
            || !assessment?.grasp_verified) {
            carryTrace.push({
              frame: snapshot.frame,
              relativePoseError: relativePoseError(
                snapshot.robot,
                carriedReferencePose
              ),
              assessment: assessment?.evidence.relative_pose,
              objectAngularVelocity:
                snapshot.robot.objects[OBJECT_ID]!.angularVelocity,
              wristAngularVelocity:
                snapshot.robot.links.left_wrist_yaw_link.angularVelocity,
              handSaturation: Object.entries(snapshot.robot.hands.joints)
                .filter(([joint, evidence]) => (
                  joint.startsWith("left_") && evidence.saturated
                ))
                .map(([joint]) => joint),
              contacts: snapshot.robot.contacts.filter((contact) => (
                contact.firstObject === OBJECT_ID
                  || contact.secondObject === OBJECT_ID
              )).map((contact) => ({
                hand: contact.firstHandLink ?? contact.secondHandLink,
                normalForce: contact.normalForce,
                position: contact.position
              }))
            });
          }
        }
      );
      expect(carried.accepted, JSON.stringify({
        reason: carried.detail.reason,
        carryTrace
      })).toBe(true);
      expect(carried.code).toBe("navigation_completed");
      expect(carried.detail.travelledDistance).toBeGreaterThanOrEqual(0.3);
      expect(carried.detail.carry).toMatchObject({
        continuation: { continued: true },
        unauthorized_contacts: []
      });
      expect(invalidCarryFrame).toBeNull();
      const objectAfterCarry = carried.finalSnapshot.robot.objects[OBJECT_ID]!.position;
      expect(Math.hypot(
        objectAfterCarry.x - objectBeforeCarry.x,
        objectAfterCarry.z - objectBeforeCarry.z
      )).toBeGreaterThanOrEqual(0.3);
      const carriedAssessment = carried.finalSnapshot.grasp.assessments.find((entry) => (
        entry.object_id === OBJECT_ID && entry.hand === "left"
      ));
      expect(carriedAssessment).toMatchObject({
        grasp_verified: true,
        reason: "grasp_verified"
      });
      const graspContract = world.checkpoint().graspRegistry.contract;
      expect(carriedAssessment!.evidence.relative_pose.translation_drift_m)
        .toBeLessThanOrEqual(graspContract.maximum_relative_translation_drift_m);
      expect(carriedAssessment!.evidence.relative_pose.rotation_drift_rad)
        .toBeLessThanOrEqual(graspContract.maximum_relative_rotation_drift_rad);

      const releasePelvis = carried.finalSnapshot.robot.links.pelvis;
      const releaseWrist = rotateVector(
        inverseQuaternion(releasePelvis.rotation),
        subtract(
          carried.finalSnapshot.robot.links.left_wrist_yaw_link.position,
          releasePelvis.position
        )
      );
      const releaseWristOrientation = normalizeQuaternion(multiplyQuaternion(
        inverseQuaternion(releasePelvis.rotation),
        carried.finalSnapshot.robot.links.left_wrist_yaw_link.rotation
      ));
      const releasePlan = (input: {
        id: string;
        extendZ: number;
        retreatForward: number;
        withdrawX: number;
        withdrawY: number;
        withdrawZ: number;
      }): HumanoidMotionPlan => ({
        id: input.id,
        intent: "主动张开左手、与物体完全分离并等待物体在目标区域落稳",
        duration_seconds: 6,
        contact_constraints: surfaces,
        keyframes: [{
          at_seconds: 0,
          root_velocity: { forward_mps: 0, lateral_mps: 0 },
          hand_coordination: CLOSED_LEFT_HAND,
          left_hand: {
            frame: "pelvis",
            position: releaseWrist,
            tolerance_m: 0.05,
            orientation: releaseWristOrientation,
            orientation_tolerance_rad: 0.15
          }
        }, {
          at_seconds: 1.2,
          root_velocity: { forward_mps: 0, lateral_mps: 0 },
          hand_coordination: CLOSED_LEFT_HAND,
          left_hand: {
            frame: "pelvis",
            position: {
              ...releaseWrist,
              y: releaseWrist.y + 0.03,
              z: releaseWrist.z + input.extendZ
            },
            tolerance_m: 0.075
          }
        }, {
          at_seconds: 1.8,
          root_velocity: {
            forward_mps: input.retreatForward,
            lateral_mps: 0
          },
          hand_coordination: OPEN_HANDS,
          left_hand: {
            frame: "pelvis",
            position: {
              ...releaseWrist,
              y: releaseWrist.y + 0.03,
              z: releaseWrist.z + input.extendZ
            },
            tolerance_m: 0.075
          }
        }, {
          at_seconds: 2.6,
          root_velocity: {
            forward_mps: input.retreatForward,
            lateral_mps: 0
          },
          hand_coordination: OPEN_HANDS,
          left_hand: {
            frame: "pelvis",
            position: {
              ...releaseWrist,
              x: releaseWrist.x + input.withdrawX,
              y: releaseWrist.y + input.withdrawY,
              z: releaseWrist.z + input.extendZ + input.withdrawZ
            },
            tolerance_m: 0.12
          }
        }, {
          at_seconds: 6,
          root_velocity: { forward_mps: 0, lateral_mps: 0 },
          hand_coordination: OPEN_HANDS,
          left_hand: {
            frame: "pelvis",
            position: {
              ...releaseWrist,
              x: releaseWrist.x + input.withdrawX,
              y: releaseWrist.y + input.withdrawY,
              z: releaseWrist.z + input.extendZ + input.withdrawZ
            },
            tolerance_m: 0.12
          }
        }]
      });
      const releaseContract = {
        option_id: "world-left-release-settle",
        predicates: [{
          type: "grasp_verified" as const,
          object_id: OBJECT_ID,
          hand: "left" as const,
          grasp_contract_sha256: before.grasp.contractSha256
        }, {
          type: "object_released" as const,
          object_id: OBJECT_ID,
          hand: "left" as const
        }, {
          type: "object_settled_on_support" as const,
          object_id: OBJECT_ID
        }, {
          type: "object_in_zone" as const,
          object_id: OBJECT_ID,
          zone_id: "drop-zone",
          expected: true,
          tolerance_m: 0.05
        }],
        stable_steps: 12,
        phases: {
          precondition: {
            condition: { op: "predicate" as const, predicate_index: 0 },
            stable_steps: 1
          },
          during: null,
          terminal: {
            condition: {
              op: "all" as const,
              conditions: [{
                op: "not" as const,
                condition: { op: "predicate" as const, predicate_index: 0 }
              }, {
                op: "predicate" as const,
                predicate_index: 1
              }, {
                op: "predicate" as const,
                predicate_index: 2
              }, {
                op: "predicate" as const,
                predicate_index: 3
              }]
            }
          }
        }
      };
      const release = await world.planWholeBodyMotionCandidates({
        objective: "在目标区域主动释放携带物并等待真实落稳",
        termination: releaseContract,
        candidates: [
          releasePlan({
            id: "release-forward-recover",
            extendZ: 0.12,
            retreatForward: -0.35,
            withdrawX: 0.12,
            withdrawY: 0.18,
            withdrawZ: -0.02
          }),
          releasePlan({
            id: "release-short-recover",
            extendZ: 0.12,
            retreatForward: -0.25,
            withdrawX: 0.09,
            withdrawY: 0.16,
            withdrawZ: -0.04
          })
        ]
      });
      expect(release.accepted, JSON.stringify(release.candidates.map((candidate) => ({
        rank: candidate.rank,
        planId: candidate.planId,
        failures: candidate.validation.failures,
        simulatedSteps: candidate.validation.evidence.simulatedSteps,
        releaseWrist,
        root: candidate.validation.finalSnapshot.rootPosition,
        pelvis: candidate.validation.finalSnapshot.links.pelvis.position,
        leftAnkle: candidate.validation.finalSnapshot.links.left_ankle_roll_link.position,
        rightAnkle: candidate.validation.finalSnapshot.links.right_ankle_roll_link.position,
        leftWrist: candidate.validation.finalSnapshot.links.left_wrist_yaw_link.position,
        object: candidate.validation.finalSnapshot.objects[OBJECT_ID],
        contacts: candidate.validation.finalSnapshot.contacts.filter((contact) => (
          contact.firstObject === OBJECT_ID || contact.secondObject === OBJECT_ID
        ))
      })))).toBe(true);
      const released = await world.executeWholeBodyMotion(release.planId);
      expect(released).toMatchObject({
        accepted: true,
        code: "motion_option_succeeded",
        detail: {
          option: {
            status: "succeeded",
            termination_reason: "physical_success"
          }
        }
      });
      expect(world.checkpoint().carriedObjectLifecycle).toMatchObject({
        phase: "released",
        active_binding_set: null,
        transition_reason: "release_completed"
      });
      const releasedRobot = released.finalSnapshot.robot;
      expect(assessHumanoidObjectReleased({
        objectId: OBJECT_ID,
        hand: "left",
        objectObservable: true,
        contacts: releasedRobot.contacts
      })).toMatchObject({
        status: "satisfied",
        reason: "object_released",
        handContactCount: 0
      });
      expect(assessHumanoidObjectSettledOnSupport({
        objectId: OBJECT_ID,
        objectObservable: true,
        snapshot: releasedRobot
      })).toMatchObject({
        status: "satisfied",
        reason: "object_settled_on_support"
      });
    } finally {
      await world.dispose();
    }
  }, 120_000);
});

function relativePoseError(
  snapshot: ReturnType<HumanoidSimulation["snapshot"]>,
  reference: {
    translation: { x: number; y: number; z: number };
    rotation: { x: number; y: number; z: number; w: number };
  }
): {
  translation: { x: number; y: number; z: number };
  rotation: { x: number; y: number; z: number; w: number };
} {
  const wrist = snapshot.links.left_wrist_yaw_link;
  const object = snapshot.objects[OBJECT_ID]!;
  const inverseWrist = inverseQuaternion(wrist.rotation);
  const currentTranslation = rotateVector(
    inverseWrist,
    subtract(object.position, wrist.position)
  );
  const currentRotation = normalizeQuaternion(multiplyQuaternion(
    inverseWrist,
    object.rotation
  ));
  return {
    translation: subtract(currentTranslation, reference.translation),
    rotation: normalizeQuaternion(multiplyQuaternion(
      currentRotation,
      inverseQuaternion(reference.rotation)
    ))
  };
}
