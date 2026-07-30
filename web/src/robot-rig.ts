import * as THREE from "three";
import { liveChannels, liveCommands } from "./active-commands";
import { addMesh, materialOf, roundedBoxGeometry, standardMaterial } from "./three-kit";
import type { WorldSnapshot } from "./types";

/**
 * The nine rigid bodies the physics engine actually simulates. Every group in
 * this file is placed from one of them, so if the world stops reporting a link
 * the rig refuses to draw rather than falling back to a pose it invented.
 */
export const ROBOT_LINK_IDS = [
  "base",
  "torso",
  "sensor_head",
  "upper_arm",
  "forearm",
  "wrist",
  "gripper",
  "left_finger",
  "right_finger"
] as const;

/**
 * Where the links meet, in the frame of the parent link. These are not styling
 * choices: they are read off the same ROBOT_SPEC the simulation uses, so the
 * hardware drawn at a joint sits exactly where the joint is. The arm mounts at
 * the torso's front centre and the sensor head on a mast behind it — draw
 * either one anywhere else and the robot falls apart on screen even though the
 * physics is fine.
 */
const MOUNT = {
  /** Shoulder pitch axis, in torso-local space: (0, 0.92 - 0.78, 0.16). */
  shoulder: { y: 0.14, z: 0.16 },
  /** Head pan axis, in torso-local space: (0, 1.32 - 0.78, -0.24). */
  headMast: { y: 0.54, z: -0.24 },
  /** Wheel centres: track width 0.64, radius 0.14, base centre 0.38 up. */
  wheel: { x: 0.32, y: -0.24, radius: 0.14 }
} as const;

/**
 * The linear guide the fingers ride. Palm hardware, finger carriages and
 * bushings all key off these numbers, so the rails, the blocks that slide on
 * them and the deck they emerge from stay on one axis no matter how the jaw
 * geometry is retuned.
 */
const GUIDE = {
  deckY: 0.045,
  z: 0.02,
  rodOffsets: [-0.042, 0.042]
} as const;

const PALETTE = {
  shell: 0xeef1f6,
  shellShadow: 0xb7c1d0,
  graphite: 0x1a1f27,
  carbon: 0x0a0d12,
  alloy: 0x8e9aab,
  glass: 0x06141a,
  cyan: 0x35e0c4,
  amber: 0xffb44d,
  violet: 0x8f7cf5
} as const;

/**
 * Channel colours are shared with the Journey view: the body channel a
 * command leases decides both the tag colour in the UI and which part of the
 * robot lights up, so a receipt can be traced to a moving part by colour alone.
 */
export const CHANNEL_COLOR: Record<string, number> = {
  base: PALETTE.cyan,
  head: PALETTE.violet,
  arm: PALETTE.amber,
  gripper: PALETTE.amber
};

export class RobotRig {
  readonly root = new THREE.Group();
  readonly #base = new THREE.Group();
  readonly #torso = new THREE.Group();
  readonly #sensorHead = new THREE.Group();
  readonly #upperArm = new THREE.Group();
  readonly #forearm = new THREE.Group();
  readonly #wrist = new THREE.Group();
  readonly #gripper = new THREE.Group();
  readonly #leftFinger = new THREE.Mesh();
  readonly #rightFinger = new THREE.Mesh();
  readonly #leftWheel = new THREE.Group();
  readonly #rightWheel = new THREE.Group();
  readonly #leftPad = new THREE.Mesh();
  readonly #rightPad = new THREE.Mesh();
  readonly #leftGuide = new THREE.Group();
  readonly #rightGuide = new THREE.Group();
  readonly #armBands: THREE.Mesh[] = [];
  readonly #irises: THREE.Mesh[] = [];
  readonly #sensorFrustum: THREE.LineSegments;
  readonly #chestRing: THREE.Mesh;
  readonly #driveStrip: THREE.Mesh;
  readonly #padIdle = standardMaterial({ color: PALETTE.carbon, roughness: 0.86, metalness: 0.05 });
  readonly #padContact = standardMaterial({
    color: PALETTE.amber,
    roughness: 0.5,
    metalness: 0.1,
    emissive: PALETTE.amber,
    emissiveIntensity: 1.6
  });

  constructor() {
    const shell = standardMaterial({ color: PALETTE.shell, roughness: 0.36, metalness: 0.05 });
    const shellShadow = standardMaterial({ color: PALETTE.shellShadow, roughness: 0.44, metalness: 0.09 });
    const graphite = standardMaterial({ color: PALETTE.graphite, roughness: 0.46, metalness: 0.45 });
    const carbon = standardMaterial({ color: PALETTE.carbon, roughness: 0.68, metalness: 0.18 });
    const alloy = standardMaterial({ color: PALETTE.alloy, roughness: 0.22, metalness: 0.94 });
    const glass = standardMaterial({ color: PALETTE.glass, roughness: 0.06, metalness: 0.6 });

    this.root.add(
      this.#base,
      this.#torso,
      this.#sensorHead,
      this.#upperArm,
      this.#forearm,
      this.#wrist,
      this.#gripper,
      this.#leftFinger,
      this.#rightFinger
    );

    this.#driveStrip = this.#buildBase(shell, graphite, carbon, alloy);
    this.#chestRing = this.#buildTorso(shell, shellShadow, graphite, alloy);
    this.#sensorFrustum = this.#buildSensorHead(shell, graphite, carbon, alloy, glass);
    this.#buildArm(shell, shellShadow, graphite, alloy);
    this.#buildGripper(shell, graphite, alloy);
  }

  /**
   * Drive base. Two differential wheels carry the whole machine, so they are
   * drawn full size in open arches rather than hidden under a skirt: their
   * rotation is the only direct read on odometry.
   */
  #buildBase(
    shell: THREE.Material,
    graphite: THREE.Material,
    carbon: THREE.Material,
    alloy: THREE.Material
  ): THREE.Mesh {
    addMesh(this.#base, roundedBoxGeometry(0.58, 0.26, 0.56, 0.05), graphite, [0, -0.24, 0]);
    addMesh(this.#base, roundedBoxGeometry(0.72, 0.26, 0.62, 0.08), shell, [0, -0.02, 0]);
    addMesh(this.#base, roundedBoxGeometry(0.56, 0.05, 0.5, 0.025), carbon, [0, 0.13, 0]);
    addMesh(this.#base, roundedBoxGeometry(0.44, 0.12, 0.42, 0.05), shell, [0, 0.18, 0]);

    // Panel seams. Hard-surface shapes read as designed rather than extruded
    // when the large faces are broken by a shadow line.
    for (const side of [-1, 1]) {
      addMesh(this.#base, roundedBoxGeometry(0.02, 0.16, 0.5, 0.008), carbon, [side * 0.361, -0.02, 0]);
    }
    addMesh(this.#base, roundedBoxGeometry(0.66, 0.02, 0.02, 0.008), carbon, [0, -0.15, 0.305]);

    // The base is the only part that can translate the whole robot, so its
    // heading is worth reading from any angle: a full-width bar, not a dot.
    const strip = addMesh(
      this.#base,
      roundedBoxGeometry(0.46, 0.045, 0.03, 0.02),
      standardMaterial({
        color: PALETTE.cyan,
        roughness: 0.3,
        metalness: 0.1,
        emissive: PALETTE.cyan,
        emissiveIntensity: 1.1
      }),
      [0, 0.02, 0.305]
    );

    const casterFork = addMesh(this.#base, roundedBoxGeometry(0.1, 0.14, 0.1, 0.03), graphite, [0, -0.28, -0.24]);
    casterFork.castShadow = true;
    const caster = addMesh(this.#base, new THREE.SphereGeometry(0.075, 20, 14), alloy, [0, -0.305, -0.24]);
    caster.scale.set(1, 0.9, 1);

    this.#leftWheel.position.set(MOUNT.wheel.x, MOUNT.wheel.y, 0);
    this.#rightWheel.position.set(-MOUNT.wheel.x, MOUNT.wheel.y, 0);
    this.#base.add(this.#leftWheel, this.#rightWheel);

    for (const side of [1, -1]) {
      const arch = addMesh(this.#base, new THREE.TorusGeometry(0.168, 0.028, 12, 30, Math.PI), graphite, [
        side * 0.335,
        MOUNT.wheel.y,
        0
      ]);
      arch.rotation.y = Math.PI / 2;
    }

    for (const wheel of [this.#leftWheel, this.#rightWheel]) {
      const tire = addMesh(wheel, new THREE.CylinderGeometry(MOUNT.wheel.radius, MOUNT.wheel.radius, 0.1, 30), carbon);
      tire.rotation.z = Math.PI / 2;
      const rim = addMesh(wheel, new THREE.CylinderGeometry(0.098, 0.098, 0.11, 26), alloy);
      rim.rotation.z = Math.PI / 2;
      const hub = addMesh(wheel, new THREE.CylinderGeometry(0.04, 0.04, 0.125, 18), graphite);
      hub.rotation.z = Math.PI / 2;

      // Wheels turn about X, so tread blocks and spokes lie in the YZ plane.
      // What they show is real odometry, not a spin animation.
      for (let index = 0; index < 14; index += 1) {
        const angle = (index / 14) * Math.PI * 2;
        const tread = addMesh(wheel, new THREE.BoxGeometry(0.104, 0.03, 0.038), graphite, [
          0,
          Math.cos(angle) * 0.135,
          Math.sin(angle) * 0.135
        ]);
        tread.rotation.x = -angle;
      }
      for (let index = 0; index < 5; index += 1) {
        const angle = (index / 5) * Math.PI * 2;
        const spoke = addMesh(wheel, new THREE.BoxGeometry(0.03, 0.09, 0.022), alloy, [
          0,
          Math.cos(angle) * 0.055,
          Math.sin(angle) * 0.055
        ]);
        spoke.rotation.x = -angle;
      }
    }
    return strip;
  }

  /**
   * Torso. It carries two mounts that have to line up with the kinematics: the
   * shoulder fork at the front, and the sensor mast at the back.
   */
  #buildTorso(
    shell: THREE.Material,
    shellShadow: THREE.Material,
    graphite: THREE.Material,
    alloy: THREE.Material
  ): THREE.Mesh {
    // Front face pulled back to z = 0.14 so the shoulder assembly at z = 0.16
    // stands proud of the column instead of being swallowed by it.
    addMesh(this.#torso, roundedBoxGeometry(0.4, 0.58, 0.32, 0.1), shell, [0, 0, -0.02]);
    addMesh(this.#torso, roundedBoxGeometry(0.3, 0.5, 0.07, 0.025), graphite, [0, -0.02, -0.185]);
    for (const side of [-1, 1]) {
      addMesh(this.#torso, roundedBoxGeometry(0.02, 0.4, 0.24, 0.008), graphite, [side * 0.201, -0.04, -0.02]);
    }

    const panel = addMesh(this.#torso, roundedBoxGeometry(0.24, 0.26, 0.03, 0.05), graphite, [0, -0.1, 0.135]);
    panel.receiveShadow = true;
    addMesh(this.#torso, roundedBoxGeometry(0.16, 0.02, 0.02, 0.008), shellShadow, [0, -0.2, 0.15]);

    const ring = addMesh(
      this.#torso,
      new THREE.TorusGeometry(0.072, 0.015, 14, 36),
      standardMaterial({
        color: PALETTE.graphite,
        roughness: 0.3,
        metalness: 0.4,
        emissive: PALETTE.cyan,
        emissiveIntensity: 0
      }),
      [0, -0.09, 0.152]
    );

    // Shoulder mount: a single boss carrying the pitch axis. The upper arm is
    // only ~0.17 across, so a wide straddling fork would read as a pair of ears
    // rather than as the housing the arm actually turns in.
    addMesh(this.#torso, roundedBoxGeometry(0.26, 0.22, 0.2, 0.07), shell, [0, MOUNT.shoulder.y, 0.08]);
    const housing = addMesh(this.#torso, new THREE.CylinderGeometry(0.105, 0.105, 0.19, 30), shellShadow, [
      0,
      MOUNT.shoulder.y,
      MOUNT.shoulder.z
    ]);
    housing.rotation.z = Math.PI / 2;
    for (const side of [-1, 1]) {
      const bearing = addMesh(this.#torso, new THREE.CylinderGeometry(0.085, 0.085, 0.025, 26), alloy, [
        side * 0.098,
        MOUNT.shoulder.y,
        MOUNT.shoulder.z
      ]);
      bearing.rotation.z = Math.PI / 2;
    }

    // Sensor mast, cantilevered behind the column to clear the arm's swing. It
    // runs up to the pan bearing directly under the head, so the head has
    // something to stand on at every pitch.
    addMesh(this.#torso, roundedBoxGeometry(0.2, 0.2, 0.22, 0.05), graphite, [0, 0.25, -0.15]);
    addMesh(this.#torso, new THREE.CylinderGeometry(0.062, 0.078, 0.14, 24), graphite, [0, 0.36, MOUNT.headMast.z]);
    // Tops out just under the head's chin at MOUNT.headMast.y - 0.12, so the
    // head sits on the mast instead of hovering above a gap.
    const panBearing = addMesh(this.#torso, new THREE.CylinderGeometry(0.078, 0.078, 0.05, 28), alloy, [
      0,
      MOUNT.headMast.y - 0.13,
      MOUNT.headMast.z
    ]);
    panBearing.castShadow = true;
    return ring;
  }

  /**
   * Sensor head. Which entities become visible is decided entirely by head yaw
   * and pitch, so the pod is built asymmetric front-to-back — a wrapped visor,
   * a brow, a chin — and never reads as a box that could be facing any way.
   */
  #buildSensorHead(
    shell: THREE.Material,
    graphite: THREE.Material,
    carbon: THREE.Material,
    alloy: THREE.Material,
    glass: THREE.Material
  ): THREE.LineSegments {
    addMesh(this.#sensorHead, roundedBoxGeometry(0.34, 0.24, 0.26, 0.1), shell, [0, 0, -0.02]);
    addMesh(this.#sensorHead, roundedBoxGeometry(0.16, 0.1, 0.1, 0.03), graphite, [0, -0.12, -0.06]);

    // A partial cylinder wrapping the front: theta = 0 is +Z in three's cylinder
    // parameterisation, so the band is centred on the head's forward axis.
    const visor = addMesh(
      this.#sensorHead,
      new THREE.CylinderGeometry(0.185, 0.185, 0.15, 36, 1, true, -1.15, 2.3),
      glass,
      [0, 0.005, 0]
    );
    visor.scale.set(1, 1, 0.82);

    for (const [y, height] of [
      [0.1, 0.03],
      [-0.09, 0.026]
    ] as const) {
      const trim = addMesh(
        this.#sensorHead,
        new THREE.CylinderGeometry(0.192, 0.192, height, 36, 1, true, -1.15, 2.3),
        alloy,
        [0, y, 0]
      );
      trim.scale.set(1, 1, 0.82);
    }

    for (const x of [-0.085, 0.085]) {
      const barrel = addMesh(this.#sensorHead, new THREE.CylinderGeometry(0.04, 0.046, 0.06, 24), carbon, [x, 0.005, 0.135]);
      barrel.rotation.x = Math.PI / 2;
      const iris = addMesh(
        this.#sensorHead,
        new THREE.TorusGeometry(0.03, 0.008, 12, 26),
        standardMaterial({
          color: PALETTE.graphite,
          roughness: 0.3,
          metalness: 0.5,
          emissive: PALETTE.violet,
          emissiveIntensity: 0.9
        }),
        [x, 0.005, 0.166]
      );
      this.#irises.push(iris);
    }

    // Depth bar between the stereo pair, and side vents to break the shell.
    addMesh(this.#sensorHead, roundedBoxGeometry(0.09, 0.028, 0.02, 0.01), graphite, [0, 0.062, 0.155]);
    for (const side of [-1, 1]) {
      addMesh(this.#sensorHead, roundedBoxGeometry(0.02, 0.09, 0.12, 0.008), graphite, [side * 0.166, -0.01, -0.03]);
    }

    const frustum = createSensorFrustum();
    this.#sensorHead.add(frustum);
    return frustum;
  }

  /** Arm: three links whose casings meet at matched hubs, so it reads as one limb. */
  #buildArm(
    shell: THREE.Material,
    shellShadow: THREE.Material,
    graphite: THREE.Material,
    alloy: THREE.Material
  ): void {
    this.#buildArmLink(this.#upperArm, 0.62, 0.085, 0.075, shell, graphite, alloy);
    this.#buildArmLink(this.#forearm, 0.55, 0.072, 0.062, shellShadow, graphite, alloy);
    this.#buildArmLink(this.#wrist, 0.18, 0.056, 0.052, shell, graphite, alloy);
  }

  /**
   * One arm link. Its long axis is local +Z with the joint at -length/2, which
   * is exactly how the forward kinematics places the link, so the casing runs
   * the full span and the end caps land on the neighbouring joints.
   */
  #buildArmLink(
    group: THREE.Group,
    length: number,
    rootRadius: number,
    tipRadius: number,
    shell: THREE.Material,
    graphite: THREE.Material,
    alloy: THREE.Material
  ): void {
    const half = length / 2;

    // A dark tube spanning the whole link. It is what shows through at a joint,
    // so two links never appear to float apart when the casings taper away.
    const core = addMesh(
      group,
      new THREE.CapsuleGeometry(tipRadius * 0.54, Math.max(0.01, length - tipRadius * 1.08), 4, 18),
      graphite
    );
    core.rotation.x = Math.PI / 2;

    // Tapered casing capped by spheres. The cap radii match the hub radii of the
    // neighbouring links, so parent tip and child root merge into one silhouette.
    const casing = addMesh(
      group,
      new THREE.CylinderGeometry(tipRadius, rootRadius, Math.max(0.02, length - rootRadius - tipRadius), 28),
      shell,
      [0, 0, (rootRadius - tipRadius) / 2]
    );
    casing.rotation.x = Math.PI / 2;
    addMesh(group, new THREE.SphereGeometry(rootRadius, 26, 18), shell, [0, 0, -half + rootRadius]);
    addMesh(group, new THREE.SphereGeometry(tipRadius, 26, 18), shell, [0, 0, half - tipRadius]);

    // The joint this link hangs from rotates about local X. Drawing the actuator
    // on that axis makes the visible hardware match the real degree of freedom.
    const actuator = addMesh(
      group,
      new THREE.CylinderGeometry(rootRadius * 1.12, rootRadius * 1.12, rootRadius * 2.3, 28),
      alloy,
      [0, 0, -half]
    );
    actuator.rotation.z = Math.PI / 2;
    for (const side of [-1, 1]) {
      const cap = addMesh(
        group,
        new THREE.TorusGeometry(rootRadius * 1.12, rootRadius * 0.17, 12, 28),
        graphite,
        [side * rootRadius * 1.16, 0, -half]
      );
      cap.rotation.y = Math.PI / 2;
    }

    // A torus lies in XY with its axis on +Z, which is the link's long axis, so
    // this band wraps the casing without any rotation.
    const band = addMesh(
      group,
      new THREE.TorusGeometry(tipRadius * 1.04, tipRadius * 0.14, 10, 28),
      standardMaterial({
        color: PALETTE.graphite,
        roughness: 0.34,
        metalness: 0.4,
        emissive: PALETTE.amber,
        emissiveIntensity: 0.2
      }),
      [0, 0, half - tipRadius * 2.4]
    );
    this.#armBands.push(band);
  }

  /**
   * Gripper. The fingers are separate rigid bodies placed by the solver at
   * +/- aperture/2, so the guide rails between palm and finger are scaled from
   * the reported aperture rather than drawn at a fixed width.
   */
  #buildGripper(shell: THREE.Material, graphite: THREE.Material, alloy: THREE.Material): void {
    // The palm is a housing tall enough to contain the rail deck, so the rails
    // emerge from a body instead of floating across the whole aperture.
    addMesh(this.#gripper, roundedBoxGeometry(0.18, 0.18, 0.13, 0.025), shell, [0, 0.015, 0.03]);
    addMesh(this.#gripper, roundedBoxGeometry(0.195, 0.08, 0.15, 0.022), graphite, [0, GUIDE.deckY, 0.02]);
    addMesh(this.#gripper, roundedBoxGeometry(0.12, 0.035, 0.04, 0.014), shell, [0, -0.06, 0.078]);

    const collar = addMesh(this.#gripper, new THREE.CylinderGeometry(0.058, 0.048, 0.08, 26), alloy, [0, 0, -0.03]);
    collar.rotation.x = -Math.PI / 2;

    for (const [guide, side] of [
      [this.#leftGuide, 1],
      [this.#rightGuide, -1]
    ] as const) {
      // Two rods per side rather than one: a single rod centred on the finger is
      // what makes a parallel gripper read as a barbell, because nothing about it
      // says which way the finger is constrained. The pair are unit length along
      // local Y and scaled each frame to the reported aperture.
      guide.rotation.z = (side * Math.PI) / 2;
      guide.position.set(0, GUIDE.deckY, GUIDE.z);
      this.#gripper.add(guide);
      for (const rod of GUIDE.rodOffsets) {
        const mesh = addMesh(guide, new THREE.CylinderGeometry(0.012, 0.012, 1, 16), alloy);
        // The guide's own rotation already lies along lateral X, so a rod is
        // offset within the deck by moving it along the guide's local Z.
        mesh.position.z = rod;
      }
    }

    for (const [finger, pad, side] of [
      [this.#leftFinger, this.#leftPad, 1],
      [this.#rightFinger, this.#rightPad, -1]
    ] as const) {
      // The outer dimensions match ROBOT_SPEC.gripper.fingerHalfExtents, so what
      // is drawn is the same volume the solver tests for contact.
      finger.geometry = roundedBoxGeometry(0.06, 0.24, 0.14, 0.022);
      finger.material = standardMaterial({ color: PALETTE.shellShadow, roughness: 0.4, metalness: 0.12 });
      finger.castShadow = true;
      finger.receiveShadow = true;

      // The carriage sits at the rails' height, not at the top of the finger: it
      // is the part that rides them, and putting it anywhere else leaves the rods
      // passing through the finger with nothing to explain the connection. It
      // wears the finger's own shell so it reads as the top of the jaw rather
      // than a separate black brick clamped onto it.
      const carriage = addMesh(finger, roundedBoxGeometry(0.075, 0.12, 0.148, 0.022), shell, [0, GUIDE.deckY, GUIDE.z]);
      carriage.castShadow = true;
      addMesh(finger, roundedBoxGeometry(0.062, 0.055, 0.1, 0.016), graphite, [0, GUIDE.deckY + 0.035, GUIDE.z]);
      for (const rod of GUIDE.rodOffsets) {
        const bushing = addMesh(finger, new THREE.CylinderGeometry(0.021, 0.021, 0.055, 20), alloy, [
          -side * 0.042,
          GUIDE.deckY,
          GUIDE.z + rod
        ]);
        bushing.rotation.z = Math.PI / 2;
      }

      // The pad faces the other finger and covers the jaw below the carriage, so
      // the contact colour appears where contact actually happens.
      pad.geometry = roundedBoxGeometry(0.014, 0.14, 0.11, 0.006);
      pad.material = this.#padIdle;
      pad.position.set(-side * 0.031, -0.055, 0.005);
      pad.castShadow = true;
      finger.add(pad);
    }
  }

  update(snapshot: WorldSnapshot): void {
    const { robot } = snapshot;
    const missing = ROBOT_LINK_IDS.filter((linkId) => robot.links?.[linkId] === undefined);
    if (missing.length > 0) {
      throw new Error(`World state is missing robot links: ${missing.join(", ")}`);
    }
    applyLinkTransform(this.#base, robot.links, "base");
    applyLinkTransform(this.#torso, robot.links, "torso");
    applyLinkTransform(this.#sensorHead, robot.links, "sensor_head");
    applyLinkTransform(this.#upperArm, robot.links, "upper_arm");
    applyLinkTransform(this.#forearm, robot.links, "forearm");
    applyLinkTransform(this.#wrist, robot.links, "wrist");
    applyLinkTransform(this.#gripper, robot.links, "gripper");
    applyLinkTransform(this.#leftFinger, robot.links, "left_finger");
    applyLinkTransform(this.#rightFinger, robot.links, "right_finger");

    this.#leftPad.material = robot.contacts.left_object_id !== null ? this.#padContact : this.#padIdle;
    this.#rightPad.material = robot.contacts.right_object_id !== null ? this.#padContact : this.#padIdle;

    // The fingers are separate rigid bodies the solver places at +/- aperture/2,
    // so the rails have to be re-spanned from the reported aperture: nothing
    // else in the scene would keep the palm and the fingers connected. Each rail
    // runs from inside the palm deck to its finger's carriage, so it is never
    // longer than the gap it actually bridges and never pokes out the far side.
    const halfAperture = Math.max(0.012, robot.gripper.aperture / 2);
    const palmEdge = 0.078;
    const span = Math.max(0.01, halfAperture - palmEdge + 0.03);
    for (const [guide, side] of [
      [this.#leftGuide, 1],
      [this.#rightGuide, -1]
    ] as const) {
      guide.scale.y = span;
      guide.position.x = side * (palmEdge + span / 2 - 0.03);
      guide.visible = span > 0.02;
    }

    this.#leftWheel.rotation.x = robot.odometry.left_wheel.position;
    this.#rightWheel.rotation.x = robot.odometry.right_wheel.position;

    const commands = liveCommands(snapshot);
    const channels = liveChannels(snapshot);
    const channel = (["gripper", "arm", "base", "head"] as const)
      .find((candidate) => channels.includes(candidate));
    const ringMaterial = materialOf(this.#chestRing);
    ringMaterial.emissive.setHex(channel ? CHANNEL_COLOR[channel] ?? PALETTE.cyan : PALETTE.cyan);
    ringMaterial.emissiveIntensity = commands.length > 0 ? 1.4 : 0.12;

    const driving = Math.abs(robot.odometry.left_wheel.velocity) + Math.abs(robot.odometry.right_wheel.velocity);
    materialOf(this.#driveStrip).emissiveIntensity = driving > 0.02 ? 1.9 : 0.55;

    const armActive = channels.some((entry) => entry === "arm" || entry === "gripper");
    for (const band of this.#armBands) {
      materialOf(band).emissiveIntensity = armActive ? 1.7 : 0.2;
    }

    const looking = robot.joint_status.head_yaw?.velocity ?? 0;
    for (const iris of this.#irises) {
      materialOf(iris).emissiveIntensity = Math.abs(looking) > 0.02 ? 1.8 : 0.7;
    }
    const frustumMaterial = this.#sensorFrustum.material as THREE.LineBasicMaterial;
    frustumMaterial.opacity = channels.includes("head") ? 0.72 : 0.2;
  }
}

/** A short, physical-FOV guide anchored to the articulated sensor link. */
function createSensorFrustum(): THREE.LineSegments {
  const origin = new THREE.Vector3(0, 0, 0.17);
  const reach = 0.58;
  const halfWidth = reach * Math.tan(Math.PI * 0.72 / 2);
  const halfHeight = reach * Math.tan(Math.PI * 0.5 / 2);
  const corners = [
    new THREE.Vector3(-halfWidth, -halfHeight, origin.z + reach),
    new THREE.Vector3(halfWidth, -halfHeight, origin.z + reach),
    new THREE.Vector3(halfWidth, halfHeight, origin.z + reach),
    new THREE.Vector3(-halfWidth, halfHeight, origin.z + reach)
  ];
  const vertices: THREE.Vector3[] = [];
  for (const corner of corners) vertices.push(origin, corner);
  for (let index = 0; index < corners.length; index += 1) {
    vertices.push(corners[index]!, corners[(index + 1) % corners.length]!);
  }
  const guide = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(vertices),
    new THREE.LineBasicMaterial({
      color: PALETTE.violet,
      transparent: true,
      opacity: 0.2,
      depthWrite: false
    })
  );
  guide.name = "sensor-field-of-view";
  return guide;
}

export function applyLinkTransform(
  object: THREE.Object3D,
  links: WorldSnapshot["robot"]["links"],
  linkId: string
): void {
  const link = links[linkId];
  if (!link) throw new Error(`Physics snapshot is missing robot link: ${linkId}`);
  object.position.set(link.position.x, link.position.y, link.position.z);
  object.quaternion.set(link.rotation.x, link.rotation.y, link.rotation.z, link.rotation.w);
}
