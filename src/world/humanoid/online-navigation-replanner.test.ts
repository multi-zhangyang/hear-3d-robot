import { describe, expect, it } from "vitest";
import {
  MAXIMUM_ONLINE_NAVIGATION_REPLANS,
  onlineNavigationReplanDecision
} from "./online-navigation-replanner.js";

describe("online humanoid navigation replanning", () => {
  it("replans geometric obstruction while preserving safety and retry bounds", () => {
    expect(onlineNavigationReplanDecision({
      reason: "environment_contact:left_hand_palm_link:block-7",
      fallen: false,
      attempts: 0
    })).toEqual({ replan: true, failure_class: "dynamic_obstruction" });
    expect(onlineNavigationReplanDecision({
      reason: "contact_while_stopping:environment_contact:pelvis:block-7",
      fallen: false,
      attempts: 1
    })).toEqual({ replan: true, failure_class: "dynamic_obstruction" });
    expect(onlineNavigationReplanDecision({
      reason: "environment_contact:left_hand_palm_link:block-7",
      fallen: false,
      attempts: MAXIMUM_ONLINE_NAVIGATION_REPLANS
    })).toEqual({ replan: false, failure_class: "budget_exhausted" });
  });

  it("returns falls and carried-object failures to semantic recovery", () => {
    expect(onlineNavigationReplanDecision({
      reason: "environment_contact:pelvis:block-7",
      fallen: true,
      attempts: 0
    })).toEqual({ replan: false, failure_class: "unsafe_state" });
    expect(onlineNavigationReplanDecision({
      reason: "carried_object_collision:parcel:solid",
      fallen: false,
      attempts: 0
    })).toEqual({ replan: false, failure_class: "semantic_recovery" });
  });
});
