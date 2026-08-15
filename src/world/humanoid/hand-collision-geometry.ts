import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  type G1HandContactSurfaceName
} from "./morphology.js";

export type G1HandCollisionMeshName = G1HandContactSurfaceName;

export const G1_HAND_COLLISION_MESH_NAMES = [
  ...G1_HAND_CONTACT_SURFACE_NAMES
] as const satisfies readonly G1HandCollisionMeshName[];

const EXPECTED_MESH_COLLIDER_COUNT = 14;

export function g1HandContactGeomName(surface: G1HandContactSurfaceName): string {
  return `g1-hand-contact-${surface}`;
}

export function g1PhysicsModelXml(source: string): string {
  assertUniqueGeomNames(source);
  const generatedNames = new Set<string>();
  let transformed = source.replace(
    /<compiler\s+([^>]*?)\/>/,
    (_match, attributes: string) => {
      if (/\bdiscardvisual=/.test(attributes)) {
        throw new Error("G1 source model already declares discardvisual");
      }
      return `<compiler ${attributes} discardvisual="true"/>`;
    }
  );
  let replacementCount = 0;
  transformed = transformed.replace(/<geom\b([^>]*)\/>/g, (element, attributes: string) => {
    if (!isG1HandColliderClass(attribute(attributes, "class"))) return element;
    const mesh = attribute(attributes, "mesh");
    if (!mesh || !isG1HandCollisionMeshName(mesh)) return element;
    const generatedName = g1HandContactGeomName(mesh);
    const existingName = attribute(attributes, "name");
    if (existingName !== null && existingName !== generatedName) {
      throw new Error(
        `G1 hand collider ${mesh} already has geom name ${existingName}; refusing to overwrite it`
      );
    }
    if (generatedNames.has(generatedName)) {
      throw new Error(`G1 hand contact geom name is not unique: ${generatedName}`);
    }
    generatedNames.add(generatedName);
    replacementCount += 1;
    return `<geom${attributes}${existingName === null ? ` name="${generatedName}"` : ""}/>`;
  });
  if (replacementCount !== EXPECTED_MESH_COLLIDER_COUNT) {
    throw new Error(
      `G1 physics model replaced ${replacementCount} hand mesh colliders; expected ${EXPECTED_MESH_COLLIDER_COUNT}`
    );
  }
  transformed = transformed.replace(/\s*<geom\b(?=[^>]*\bclass="visual")[^>]*\/>/g, "");
  assertUniqueGeomNames(transformed);
  return transformed;
}

function isG1HandColliderClass(value: string | null): boolean {
  return value === "collision" || value === "hand_collision";
}

function isG1HandCollisionMeshName(value: string): value is G1HandCollisionMeshName {
  return (G1_HAND_COLLISION_MESH_NAMES as readonly string[]).includes(value);
}

function attribute(attributes: string, name: string): string | null {
  return new RegExp(`\\b${name}="([^"]*)"`).exec(attributes)?.[1] ?? null;
}

function assertUniqueGeomNames(source: string): void {
  const names = new Set<string>();
  for (const match of source.matchAll(/<geom\b([^>]*)\/?\s*>/g)) {
    const name = attribute(match[1] ?? "", "name");
    if (name === null) continue;
    if (names.has(name)) throw new Error(`G1 geom name is not unique: ${name}`);
    names.add(name);
  }
}
