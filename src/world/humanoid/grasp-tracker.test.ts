import { describe, expect, it } from "vitest";
import type { Vec3 } from "../../domain/schema.js";
import {
  HumanoidGraspObservationSchema,
  HumanoidGraspTracker,
  HumanoidGraspTrackerCheckpointSchema,
  humanoidGraspObservation,
  type HumanoidGraspContract,
  type HumanoidGraspHand,
  type HumanoidGraspObservation
} from "./grasp-tracker.js";
import type { G1HandContactSurfaceName } from "./morphology.js";
import type {
  HumanoidContactSnapshot,
  HumanoidSimulationSnapshot
} from "./simulation.js";

const CONTRACT = {
  protocol: "humanoid-grasp-contract-v1",
  world_up: vector(0, 1, 0),
  minimum_distinct_contact_links: 2,
  minimum_contact_normal_force_n: 5,
  maximum_opposing_normal_dot: -0.5,
  maximum_opposing_position_dot: -0.5,
  minimum_opposing_contact_separation_m: 0.02,
  minimum_contact_radial_distance_m: 0.005,
  maximum_relative_translation_drift_m: 0.01,
  maximum_relative_rotation_drift_rad: 0.05,
  minimum_relative_pose_stable_frames: 3,
  minimum_lift_m: 0.05,
  minimum_lifted_hold_frames: 2,
  minimum_support_normal_force_n: 2,
  minimum_support_up_dot: 0.7
} as const satisfies HumanoidGraspContract;

describe("HumanoidGraspTracker", () => {
  it("verifies a grasp only after opposed links, stable relative pose, lift, and hold", () => {
    const tracker = new HumanoidGraspTracker(CONTRACT);

    expect(tracker.observe(observation(0, { supported: true }))).toMatchObject({
      phase: "stabilizing",
      grasp_verified: false,
      reason: "relative_pose_stabilizing",
      evidence: {
        contact: {
          status: "opposed",
          distinct_force_qualified_links: [
            "left_hand_index_1_link",
            "left_hand_thumb_2_link"
          ],
          opposing_pair: {
            separation_m: 0.06,
            normal_dot: -1,
            position_dot: -1
          }
        },
        support: { status: "supported" },
        relative_pose: { stable_frames: 1 },
        lifted_hold_frames: 0
      }
    });
    tracker.observe(observation(1, { supported: true }));
    const supported = tracker.observe(observation(2, { supported: true }));
    expect(supported).toMatchObject({
      phase: "lifting",
      grasp_verified: false,
      reason: "object_supported",
      evidence: { relative_pose: { stable_frames: 3 } }
    });

    const lifted = tracker.observe(observation(3, {
      objectPosition: vector(0, 0.56, 0),
      supported: false
    }));
    expect(lifted).toMatchObject({
      phase: "holding",
      grasp_verified: false,
      reason: "lift_hold_incomplete",
      evidence: {
        support: { status: "unsupported" },
        lifted_hold_frames: 1
      }
    });
    expect(lifted.evidence.support.lift_m).toBeCloseTo(0.06);

    const verified = tracker.observe(observation(4, {
      objectPosition: vector(0, 0.56, 0),
      supported: false
    }));
    expect(verified).toMatchObject({
      phase: "verified",
      grasp_verified: true,
      reason: "grasp_verified",
      reset_reason: null,
      evidence: {
        relative_pose: { stable_frames: 5 },
        lifted_hold_frames: 2
      }
    });
  });

  it("tracks each object and hand independently", () => {
    const tracker = new HumanoidGraspTracker(CONTRACT);
    tracker.observe(observation(0, { objectId: "crate", hand: "left" }));
    tracker.observe(observation(0, { objectId: "crate", hand: "right" }));
    tracker.observe(observation(0, { objectId: "parcel", hand: "left" }));

    expect(tracker.checkpoint().tracks.map(({ object_id, hand }) => [object_id, hand]))
      .toEqual([
        ["crate", "left"],
        ["crate", "right"],
        ["parcel", "left"]
      ]);
    expect(tracker.observe(observation(1, {
      objectId: "crate",
      hand: "right"
    })).evidence.relative_pose.stable_frames).toBe(2);
    expect(tracker.observe(observation(1, {
      objectId: "parcel",
      hand: "left"
    })).evidence.relative_pose.stable_frames).toBe(2);
  });

  it("reports missing contact normals as insufficient evidence", () => {
    const tracker = new HumanoidGraspTracker(CONTRACT);
    const result = tracker.observe(observation(0, {
      contactOverrides: { secondNormal: null }
    }));

    expect(result).toMatchObject({
      phase: "idle",
      grasp_verified: false,
      reason: "contact_normal_insufficient",
      evidence: {
        contact: {
          status: "insufficient_normal",
          distinct_force_qualified_links: [
            "left_hand_index_1_link",
            "left_hand_thumb_2_link"
          ],
          distinct_normal_qualified_links: ["left_hand_thumb_2_link"],
          opposing_pair: null
        }
      }
    });
  });

  it("aggregates a split MuJoCo contact manifold before applying the force threshold", () => {
    const split = observation(0);
    split.hand_contacts = split.hand_contacts.flatMap((contact) => ([{
      ...contact,
      position: {
        ...contact.position,
        y: contact.position.y - 0.001
      },
      normal_force_n: 3
    }, {
      ...contact,
      position: {
        ...contact.position,
        y: contact.position.y + 0.001
      },
      normal_force_n: 3
    }]));

    expect(new HumanoidGraspTracker(CONTRACT).observe(split)).toMatchObject({
      evidence: {
        contact: {
          status: "opposed",
          observed_contact_count: 4,
          force_qualified_contact_count: 2,
          distinct_force_qualified_links: [
            "left_hand_index_1_link",
            "left_hand_thumb_2_link"
          ],
          opposing_pair: {
            first_normal_force_n: 6,
            second_normal_force_n: 6,
            separation_m: 0.06,
            normal_dot: -1,
            position_dot: -1
          }
        }
      }
    });
  });

  it("preserves edge-direction evidence without letting one strong manifold point dominate", () => {
    const robust = observation(0);
    robust.hand_contacts = [{
      hand_link: "left_hand_thumb_2_link",
      position: vector(-0.05, 0.5, 0),
      normal_from_hand: vector(1, 0, 0),
      normal_force_n: 36
    }, {
      hand_link: "left_hand_thumb_2_link",
      position: vector(-0.01, 0.5, 0),
      normal_from_hand: vector(0, 1, 0),
      normal_force_n: 100
    }, {
      hand_link: "left_hand_index_1_link",
      position: vector(0.05, 0.5, 0),
      normal_from_hand: vector(-1, 0, 0),
      normal_force_n: 20
    }];

    expect(new HumanoidGraspTracker(CONTRACT).observe(robust)).toMatchObject({
      evidence: {
        contact: {
          status: "opposed",
          observed_contact_count: 3,
          force_qualified_contact_count: 2,
          opposing_pair: {
            normal_dot: expect.closeTo(-6 / Math.sqrt(136), 12),
            position_dot: -1
          }
        }
      }
    });
  });

  it("rejects duplicate-link and non-opposed contact geometry", () => {
    const duplicateTracker = new HumanoidGraspTracker(CONTRACT);
    const duplicate = duplicateTracker.observe(observation(0, {
      contactOverrides: { secondLink: "left_hand_thumb_2_link" }
    }));
    expect(duplicate).toMatchObject({
      reason: "contact_links_insufficient",
      grasp_verified: false,
      evidence: { contact: { status: "insufficient_links" } }
    });

    const alignedTracker = new HumanoidGraspTracker(CONTRACT);
    const aligned = alignedTracker.observe(observation(0, {
      contactOverrides: { secondNormal: vector(1, 0, 0) }
    }));
    expect(aligned).toMatchObject({
      reason: "contacts_not_opposed",
      grasp_verified: false,
      evidence: { contact: { status: "not_opposed", opposing_pair: null } }
    });

    const centeredTracker = new HumanoidGraspTracker(CONTRACT);
    const centered = centeredTracker.observe(observation(0, {
      contactOverrides: {
        firstPosition: vector(0, 0.5, 0),
        secondPosition: vector(0, 0.5, 0)
      }
    }));
    expect(centered).toMatchObject({
      reason: "contact_geometry_insufficient",
      grasp_verified: false,
      evidence: { contact: { status: "insufficient_geometry" } }
    });
  });

  it("resets immediately on relative slip, contact loss, and frame gaps", () => {
    const slipTracker = new HumanoidGraspTracker(CONTRACT);
    slipTracker.observe(observation(0));
    slipTracker.observe(observation(1));
    const slipped = slipTracker.observe(observation(2, {
      objectPosition: vector(0.02, 0.5, 0),
      handPosition: vector(-0.1, 0.5, 0)
    }));
    expect(slipped).toMatchObject({
      grasp_verified: false,
      reset_reason: "relative_pose_unstable",
      reason: "relative_pose_unstable",
      evidence: {
        relative_pose: {
          stable_frames: 1
        }
      }
    });
    expect(slipped.evidence.relative_pose.translation_drift_m).toBeCloseTo(0.02);

    const contactTracker = new HumanoidGraspTracker(CONTRACT);
    contactTracker.observe(observation(0));
    const lost = contactTracker.observe(observation(1, { contacts: false }));
    expect(lost).toMatchObject({
      phase: "idle",
      grasp_verified: false,
      reason: "contact_missing",
      reset_reason: "contact_lost",
      evidence: { relative_pose: { stable_frames: 0 } }
    });

    const gapTracker = new HumanoidGraspTracker(CONTRACT);
    gapTracker.observe(observation(4));
    const gap = gapTracker.observe(observation(6));
    expect(gap).toMatchObject({
      phase: "stabilizing",
      grasp_verified: false,
      reason: "frame_discontinuity",
      reset_reason: "frame_discontinuity",
      evidence: { relative_pose: { stable_frames: 1 } }
    });
  });

  it("resets a lifted attempt as soon as the object returns to support", () => {
    const tracker = new HumanoidGraspTracker(CONTRACT);
    tracker.observe(observation(0, { supported: true }));
    tracker.observe(observation(1, { supported: true }));
    tracker.observe(observation(2, { supported: true }));
    tracker.observe(observation(3, {
      objectPosition: vector(0, 0.56, 0),
      supported: false
    }));

    const returned = tracker.observe(observation(4, {
      objectPosition: vector(0, 0.5, 0),
      supported: true
    }));
    expect(returned).toMatchObject({
      grasp_verified: false,
      reason: "support_returned",
      reset_reason: "support_returned",
      evidence: {
        support: {
          status: "supported",
          baseline_projection_m: 0.5,
          lift_m: 0
        },
        relative_pose: { stable_frames: 1 },
        lifted_hold_frames: 0
      }
    });
  });

  it("keeps a proven grasp verified while it is lowered onto support", () => {
    const tracker = new HumanoidGraspTracker(CONTRACT);
    tracker.observe(observation(0, { supported: true }));
    tracker.observe(observation(1, { supported: true }));
    tracker.observe(observation(2, {
      objectPosition: vector(0, 0.56, 0),
      supported: false
    }));
    const verified = tracker.observe(observation(3, {
      objectPosition: vector(0, 0.56, 0),
      supported: false
    }));
    expect(verified.grasp_verified).toBe(true);

    const lowered = tracker.observe(observation(4, {
      objectPosition: vector(0, 0.5, 0),
      supported: true
    }));
    expect(lowered).toMatchObject({
      phase: "verified",
      grasp_verified: true,
      reason: "grasp_verified",
      reset_reason: null,
      evidence: {
        relative_pose: { stable_frames: 5 },
        lifted_hold_frames: 2
      }
    });

    const released = tracker.observe(observation(5, {
      objectPosition: vector(0, 0.5, 0),
      supported: true,
      contacts: false
    }));
    expect(released).toMatchObject({
      phase: "idle",
      grasp_verified: false,
      reason: "contact_missing",
      reset_reason: "contact_lost"
    });
  });

  it("requires a measured support baseline and sufficient support normals", () => {
    const noBaseline = new HumanoidGraspTracker(CONTRACT);
    noBaseline.observe(observation(0, { supported: false }));
    noBaseline.observe(observation(1, { supported: false }));
    const unprovenLift = noBaseline.observe(observation(2, {
      objectPosition: vector(0, 0.8, 0),
      supported: false
    }));
    expect(unprovenLift).toMatchObject({
      grasp_verified: false,
      reason: "support_baseline_missing"
    });
    const stabilizedWithoutBaseline = noBaseline.observe(observation(3, {
      objectPosition: vector(0, 0.8, 0),
      supported: false
    }));
    noBaseline.observe(observation(4, {
      objectPosition: vector(0, 0.8, 0),
      supported: false
    }));
    const stillUnproven = noBaseline.observe(observation(5, {
      objectPosition: vector(0, 0.8, 0),
      supported: false
    }));
    expect(stabilizedWithoutBaseline.grasp_verified).toBe(false);
    expect(stillUnproven).toMatchObject({
      grasp_verified: false,
      reason: "support_baseline_missing",
      evidence: { support: { baseline_projection_m: null, lift_m: null } }
    });

    const unknownSupport = new HumanoidGraspTracker(CONTRACT);
    unknownSupport.observe(observation(0));
    const insufficient = unknownSupport.observe(observation(1, {
      supportNormal: null
    }));
    expect(insufficient).toMatchObject({
      phase: "idle",
      grasp_verified: false,
      reason: "support_evidence_insufficient",
      reset_reason: "support_evidence_insufficient",
      evidence: { support: { status: "insufficient_normal" } }
    });
  });

  it("aggregates split upward support forces without counting other directions", () => {
    const tracker = new HumanoidGraspTracker(CONTRACT);
    const splitSupport = observation(0);
    splitSupport.support_contacts = [
      supportContact(0.55, vector(0, 1, 0)),
      supportContact(0.55, vector(0, 1, 0)),
      supportContact(0.55, vector(0, 1, 0)),
      supportContact(0.55, vector(0, 1, 0)),
      supportContact(20, vector(0, -1, 0))
    ];

    expect(tracker.observe(splitSupport)).toMatchObject({
      evidence: {
        support: {
          status: "supported",
          candidate_contact_count: 5,
          force_qualified_contact_count: 4,
          upward_contact_count: 4
        }
      }
    });

    const downwardOnly = observation(0);
    downwardOnly.support_contacts = [
      supportContact(1.9, vector(0, 1, 0)),
      supportContact(20, vector(0, -1, 0))
    ];
    expect(new HumanoidGraspTracker(CONTRACT).observe(downwardOnly)).toMatchObject({
      evidence: {
        support: {
          status: "unsupported",
          force_qualified_contact_count: 0,
          upward_contact_count: 1
        }
      }
    });
  });

  it("serializes deterministically, resumes progress, and rejects corrupt restoration", () => {
    const tracker = new HumanoidGraspTracker(CONTRACT);
    tracker.observe(observation(0, { supported: true }));
    tracker.observe(observation(1, { supported: true }));
    tracker.observe(observation(2, { supported: true }));
    tracker.observe(observation(3, {
      objectPosition: vector(0, 0.56, 0),
      supported: false
    }));
    const checkpoint = tracker.checkpoint();
    expect(HumanoidGraspTrackerCheckpointSchema.parse(checkpoint)).toEqual(checkpoint);

    const restored = new HumanoidGraspTracker(CONTRACT, checkpoint);
    expect(restored.checkpoint()).toEqual(checkpoint);
    expect(restored.observe(observation(4, {
      objectPosition: vector(0, 0.56, 0),
      supported: false
    })).grasp_verified).toBe(true);

    expect(() => new HumanoidGraspTracker({
      ...CONTRACT,
      minimum_lift_m: 0.08
    }, checkpoint)).toThrow(/contract does not match/);

    const inconsistent = structuredClone(checkpoint);
    inconsistent.tracks[0]!.attempt!.stable_frames += 1;
    expect(() => new HumanoidGraspTracker(CONTRACT, inconsistent)).toThrow(
      /Stable grasp frames must be consecutive/
    );

    expect(() => HumanoidGraspTrackerCheckpointSchema.parse({
      ...checkpoint,
      tracks: [...checkpoint.tracks, structuredClone(checkpoint.tracks[0]!)]
    })).toThrow(/duplicate object-hand track/);
    expect(() => HumanoidGraspTrackerCheckpointSchema.parse({
      ...checkpoint,
      undocumented: true
    })).toThrow();
  });

  it("preserves exact palm and finger surfaces without inferring a palm from wrist contact", () => {
    const snapshot = minimalSnapshot([
      contact({
        firstHandLink: "left_hand_palm_link",
        secondObject: "crate",
        position: vector(0, 0.5, -0.03),
        normal: vector(0, 0, 1),
        normalForce: 9
      }),
      contact({
        firstHandLink: "left_hand_thumb_2_link",
        secondObject: "crate",
        position: vector(-0.03, 0.5, 0),
        normal: vector(2, 0, 0),
        normalForce: 10
      }),
      contact({
        firstObject: "crate",
        secondHandLink: "left_hand_index_1_link",
        position: vector(0.03, 0.5, 0),
        normal: vector(2, 0, 0),
        normalForce: 11
      }),
      contact({
        firstObject: "crate",
        position: vector(0, 0.47, 0),
        normal: vector(0, -3, 0),
        normalForce: 15
      }),
      contact({
        firstBody: "left_wrist_yaw_link",
        secondObject: "crate",
        position: vector(0, 0.5, 0),
        normal: vector(0, 1, 0),
        normalForce: 4
      })
    ]);

    const built = humanoidGraspObservation({
      frame: 7,
      objectId: "crate",
      hand: "left",
      snapshot
    });
    expect(built.hand_contacts).toEqual([
      expect.objectContaining({
        hand_link: "left_hand_palm_link",
        normal_from_hand: vector(0, 0, 1)
      }),
      expect.objectContaining({
        hand_link: "left_hand_thumb_2_link",
        normal_from_hand: vector(1, 0, 0)
      }),
      expect.objectContaining({
        hand_link: "left_hand_index_1_link",
        normal_from_hand: vector(-1, 0, 0)
      })
    ]);
    expect(built.support_contacts).toEqual([
      expect.objectContaining({
        counterpart_kind: "environment",
        normal_toward_object: vector(0, 1, 0)
      }),
      expect.objectContaining({
        counterpart_kind: "humanoid",
        counterpart_id: "left_wrist_yaw_link"
      })
    ]);
    expect(built.hand_contacts).toHaveLength(3);

    const missingNormalSnapshot = minimalSnapshot([
      contact({
        firstHandLink: "left_hand_thumb_2_link",
        secondObject: "crate",
        normal: vector(0, 0, 0),
        normalForce: 10
      })
    ]);
    expect(humanoidGraspObservation({
      frame: 0,
      objectId: "crate",
      hand: "left",
      snapshot: missingNormalSnapshot
    }).hand_contacts[0]!.normal_from_hand).toBeNull();
  });

  it("rejects observations whose contact links belong to the other hand", () => {
    expect(() => HumanoidGraspObservationSchema.parse(observation(0, {
      hand: "left",
      contactOverrides: {
        firstLink: "right_hand_thumb_2_link",
        secondLink: "right_hand_index_1_link"
      }
    }))).toThrow(/does not belong to the tracked hand/);
  });
});

interface ObservationOptions {
  objectId?: string;
  hand?: HumanoidGraspHand;
  objectPosition?: Vec3;
  handPosition?: Vec3;
  supported?: boolean;
  supportNormal?: Vec3 | null;
  contacts?: boolean;
  contactOverrides?: {
    firstLink?: G1HandContactSurfaceName;
    secondLink?: G1HandContactSurfaceName;
    firstPosition?: Vec3;
    secondPosition?: Vec3;
    firstNormal?: Vec3 | null;
    secondNormal?: Vec3 | null;
  };
}

function observation(
  frame: number,
  options: ObservationOptions = {}
): HumanoidGraspObservation {
  const hand = options.hand ?? "left";
  const objectPosition = options.objectPosition ?? vector(0, 0.5, 0);
  const handPosition = options.handPosition ?? vector(-0.1, objectPosition.y, 0);
  const sideLinks = hand === "left"
    ? ["left_hand_thumb_2_link", "left_hand_index_1_link"] as const
    : ["right_hand_thumb_2_link", "right_hand_index_1_link"] as const;
  const contactOverrides = options.contactOverrides ?? {};
  const translate = (point: Vec3): Vec3 => ({
    x: point.x + objectPosition.x,
    y: point.y + objectPosition.y,
    z: point.z + objectPosition.z
  });
  return HumanoidGraspObservationSchema.parse({
    protocol: "humanoid-grasp-observation-v1",
    frame,
    object_id: options.objectId ?? "crate",
    hand,
    object_pose: { position: objectPosition, rotation: quaternion() },
    hand_anchor_pose: { position: handPosition, rotation: quaternion() },
    hand_contacts: options.contacts === false ? [] : [{
      hand_link: contactOverrides.firstLink ?? sideLinks[0],
      position: contactOverrides.firstPosition ?? translate(vector(-0.03, 0, 0)),
      normal_from_hand: contactOverrides.firstNormal === undefined
        ? vector(1, 0, 0)
        : contactOverrides.firstNormal,
      normal_force_n: 10
    }, {
      hand_link: contactOverrides.secondLink ?? sideLinks[1],
      position: contactOverrides.secondPosition ?? translate(vector(0.03, 0, 0)),
      normal_from_hand: contactOverrides.secondNormal === undefined
        ? vector(-1, 0, 0)
        : contactOverrides.secondNormal,
      normal_force_n: 11
    }],
    support_contacts: options.supported === false ? [] : [{
      position: translate(vector(0, -0.03, 0)),
      normal_toward_object: options.supportNormal === undefined
        ? vector(0, 1, 0)
        : options.supportNormal,
      normal_force_n: 15,
      counterpart_kind: "environment",
      counterpart_id: null
    }]
  });
}

function minimalSnapshot(
  contacts: HumanoidContactSnapshot[]
): HumanoidSimulationSnapshot {
  const link = {
    position: vector(-0.1, 0.5, 0),
    rotation: quaternion(),
    linearVelocity: vector(0, 0, 0),
    angularVelocity: vector(0, 0, 0)
  };
  return {
    objects: {
      crate: {
        id: "crate",
        position: vector(0, 0.5, 0),
        rotation: quaternion(),
        linearVelocity: vector(0, 0, 0),
        angularVelocity: vector(0, 0, 0)
      }
    },
    links: {
      left_wrist_yaw_link: link,
      right_wrist_yaw_link: link
    },
    contacts
  } as unknown as HumanoidSimulationSnapshot;
}

function contact(
  overrides: Partial<HumanoidContactSnapshot>
): HumanoidContactSnapshot {
  return {
    position: vector(0, 0.5, 0),
    normal: vector(0, 1, 0),
    normalForce: 0,
    firstBody: null,
    secondBody: null,
    firstObject: null,
    secondObject: null,
    firstHandLink: null,
    secondHandLink: null,
    ...overrides
  };
}

function vector(x: number, y: number, z: number): Vec3 {
  return { x, y, z };
}

function quaternion(): { x: number; y: number; z: number; w: number } {
  return { x: 0, y: 0, z: 0, w: 1 };
}

function supportContact(
  normalForce: number,
  normal: Vec3 | null
): HumanoidGraspObservation["support_contacts"][number] {
  return {
    position: vector(0, 0.47, 0),
    normal_toward_object: normal,
    normal_force_n: normalForce,
    counterpart_kind: "environment",
    counterpart_id: null
  };
}
