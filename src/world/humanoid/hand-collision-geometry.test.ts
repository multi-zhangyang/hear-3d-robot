import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import {
  G1_HAND_COLLISION_MESH_NAMES,
  g1HandContactGeomName,
  g1PhysicsModelXml,
  type G1HandCollisionMeshName
} from "./hand-collision-geometry.js";

describe("G1 hand collision identity", () => {
  it("names every native MuJoCo mesh collider without replacing its geometry", async () => {
    const source = await sourceModel();
    const transformed = g1PhysicsModelXml(source);
    const sourceSurfaces = [...source.matchAll(/<geom\b([^>]*)\/>/g)]
      .map((match) => match[1] ?? "")
      .filter((attributes) => attribute(attributes, "class") === "collision")
      .map((attributes) => attribute(attributes, "mesh"))
      .filter((mesh): mesh is G1HandCollisionMeshName => (
        mesh !== null
          && (G1_HAND_COLLISION_MESH_NAMES as readonly string[]).includes(mesh)
      ));
    const generated = [...transformed.matchAll(/<geom\b([^>]*)\/>/g)]
      .map((match) => match[1] ?? "")
      .filter((attributes) => (
        attribute(attributes, "name")?.startsWith("g1-hand-contact-") ?? false
      ));
    const generatedNames = generated.map((attributes) => attribute(attributes, "name")!);

    expect(sourceSurfaces).toHaveLength(14);
    expect(generatedNames).toHaveLength(sourceSurfaces.length);
    expect(new Set(generatedNames).size).toBe(generatedNames.length);
    expect(generatedNames.sort()).toEqual(
      sourceSurfaces.map(g1HandContactGeomName).sort()
    );
    expect(generated.every((attributes) => (
      attribute(attributes, "type") === null
        && attribute(attributes, "mesh") !== null
        && attribute(attributes, "friction") === null
    ))).toBe(true);
    expect(transformed).toContain('<mesh file="left_hand_palm_link.STL"/>');
    expect(transformed).toContain('<mesh file="right_hand_palm_link.STL"/>');
    expect(generatedNames).toContain(g1HandContactGeomName("left_hand_palm_link"));
    expect(generatedNames).toContain(g1HandContactGeomName("right_hand_palm_link"));
  });

  it("rejects existing or duplicate geom names instead of overwriting them", async () => {
    const source = await sourceModel();
    const namedPalm = source.replace(
      /<geom([^>]*\bclass="collision"[^>]*\bmesh="left_hand_palm_link"[^>]*)\/>/,
      '<geom$1 name="upstream-left-palm"/>'
    );
    expect(namedPalm).not.toBe(source);
    expect(() => g1PhysicsModelXml(namedPalm)).toThrow(/refusing to overwrite/);

    const duplicateNames = source.replace(
      "</worldbody>",
      '<geom name="duplicate-contact-name"/><geom name="duplicate-contact-name"/></worldbody>'
    );
    expect(() => g1PhysicsModelXml(duplicateNames)).toThrow(
      /geom name is not unique/
    );
  });
});

async function sourceModel(): Promise<string> {
  return readFile(new URL(
    "../../../assets/humanoid/g1/g1_with_hands.xml",
    import.meta.url
  ), "utf8");
}

function attribute(attributes: string, name: string): string | null {
  return new RegExp(`\\b${name}="([^"]*)"`).exec(attributes)?.[1] ?? null;
}
