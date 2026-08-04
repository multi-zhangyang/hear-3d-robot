import { describe, expect, it } from "vitest";
import { assessHumanoidObjectReleased } from "./object-release.js";

describe("physical object release", () => {
  it("requires observable, complete separation from the selected hand", () => {
    const contacts = [{
      normalForce: 3.5,
      firstObject: "parcel",
      secondObject: null,
      firstHandLink: null,
      secondHandLink: "left_hand_index_1_link"
    }, {
      normalForce: 7,
      firstObject: null,
      secondObject: "parcel",
      firstHandLink: "right_hand_thumb_2_link",
      secondHandLink: null
    }];

    expect(assessHumanoidObjectReleased({
      objectId: "parcel",
      hand: "left",
      objectObservable: true,
      contacts
    })).toEqual({
      protocol: "humanoid-object-release-assessment-v1",
      objectId: "parcel",
      hand: "left",
      status: "unsatisfied",
      reason: "hand_contact_present",
      objectObservable: true,
      handContactCount: 1,
      contactSurfaces: ["left_hand_index_1_link"],
      totalNormalForceN: 3.5
    });
    expect(assessHumanoidObjectReleased({
      objectId: "parcel",
      hand: "left",
      objectObservable: true,
      contacts: contacts.slice(1)
    })).toMatchObject({
      status: "satisfied",
      reason: "object_released",
      handContactCount: 0,
      totalNormalForceN: 0
    });
    expect(assessHumanoidObjectReleased({
      objectId: "parcel",
      hand: "left",
      objectObservable: false,
      contacts: []
    })).toMatchObject({
      status: "uncertain",
      reason: "object_not_observable",
      handContactCount: null,
      totalNormalForceN: null
    });
  });
});
