import { OrbitControls } from "@react-three/drei/core/OrbitControls";
import { Canvas, extend, useFrame, useThree } from "@react-three/fiber";
import { memo, useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three/webgpu";
import type { HumanoidFrameBuffer } from "../stage/humanoid-frame-buffer";
import { renderPixelRatio, SHADOW_MAP_SIZE } from "../stage/render-quality";
import type {
  HumanoidRunDetails,
  HumanoidWorldSnapshot,
  ScenarioChunkDeltaState,
  ScenarioDefinition
} from "../types";
import { useHumanoidHudFrame } from "./use-humanoid-frame";
import { HumanoidWorldScene } from "./humanoid-world-scene";

extend(THREE as never);

export type HumanoidCameraMode = "follow" | "world" | "head";
type OrbitControlsImpl = React.ElementRef<typeof OrbitControls>;

interface HumanoidStageProps {
  details: HumanoidRunDetails;
  frameBuffer: HumanoidFrameBuffer;
  live: boolean;
}

function HumanoidStageComponent(props: HumanoidStageProps): React.JSX.Element {
  const [mode, setMode] = useState<HumanoidCameraMode>("follow");
  const [loading, setLoading] = useState(true);
  const [failure, setFailure] = useState(false);
  const [fitRevision, setFitRevision] = useState(0);
  const [backend, setBackend] = useState<"webgpu" | "webgl2" | "loading">("loading");
  const frame = useHumanoidHudFrame(props.frameBuffer, props.details.checkpoint.world);
  const extent = Math.max(
    props.details.definition.scenario.bounds.width,
    props.details.definition.scenario.bounds.depth
  );
  const markReady = useCallback(() => {
    setLoading(false);
    setFailure(false);
  }, []);
  const markFailure = useCallback(() => {
    setLoading(false);
    setFailure(true);
  }, []);

  return (
    <section className="humanoid-stage" aria-label={props.live ? "实时人形世界" : "人形世界回顾"}>
      <div className="three-stage">
        <Canvas
          className="three-canvas humanoid-canvas-shell"
          frameloop="demand"
          shadows
          dpr={renderPixelRatio(
            window.innerWidth,
            window.innerHeight,
            window.devicePixelRatio || 1
          )}
          camera={{ fov: 42, near: 0.03, far: Math.max(140, extent * 5) }}
          gl={async (rendererProps) => {
            const renderer = new THREE.WebGPURenderer({
              ...rendererProps,
              antialias: true,
              alpha: false,
              powerPreference: "high-performance"
            } as ConstructorParameters<typeof THREE.WebGPURenderer>[0]);
            renderer.outputColorSpace = THREE.SRGBColorSpace;
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1.08;
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFShadowMap;
            const inheritedDeviceLost = renderer.onDeviceLost.bind(renderer);
            renderer.onDeviceLost = (info) => {
              inheritedDeviceLost(info);
              markFailure();
            };
            await renderer.init();
            renderer.domElement.dataset.renderBackend = "isWebGPUBackend" in renderer.backend
              && renderer.backend.isWebGPUBackend === true
              ? "webgpu"
              : "webgl2";
            return renderer;
          }}
          onCreated={({ gl }) => {
            gl.domElement.classList.add("humanoid-canvas");
            const kind = gl.domElement.dataset.renderBackend;
            setBackend(kind === "webgpu" ? "webgpu" : "webgl2");
          }}
        >
          <StageScene
            runId={props.details.definition.run_id}
            scenario={props.details.definition.scenario}
            scenarioChunks={props.details.scenario_chunks}
            initialFrame={props.details.checkpoint.world}
            frameBuffer={props.frameBuffer}
            live={props.live}
            mode={mode}
            fitRevision={fitRevision}
            onReady={markReady}
            onFailure={markFailure}
          />
        </Canvas>
        {loading && <div className="stage-loading" role="status" aria-label="正在载入 G1"><i /><i /><i /></div>}
        {failure && <div className="graphics-error" role="alert">三维人形场景不可用</div>}
        <div className="humanoid-stage-bar">
          <div className="camera-switch" role="group" aria-label="观察视角">
            {(["follow", "world", "head"] as const).map((value) => (
              <button
                type="button"
                key={value}
                className={mode === value ? "active" : ""}
                aria-pressed={mode === value}
                onClick={() => setMode(value)}
              >
                {value === "follow" ? "跟随" : value === "world" ? "世界" : "头部"}
              </button>
            ))}
          </div>
          <span className={`render-backend ${backend}`}>{backend === "loading" ? "GPU…" : backend.toUpperCase()}</span>
          <button type="button" className="fit-world" onClick={() => setFitRevision((value) => value + 1)}>复位视角</button>
        </div>
        <div className={`humanoid-physics-strip ${frame.robot.fallen ? "danger" : ""}`}>
          <span><small>帧</small><b>{frame.frame.toLocaleString("zh-CN")}</b></span>
          <span><small>支撑</small><b>{supportLabel(frame.robot.balance.support)}</b></span>
          <span><small>接触</small><b>{frame.robot.contactCount}</b></span>
          <span><small>区块</small><b>R{props.details.scenario_chunks.revision}</b></span>
          <span><small>直立</small><b>{Math.round(frame.robot.balance.upright * 100)}%</b></span>
        </div>
      </div>
    </section>
  );
}

interface StageSceneProps {
  runId: string;
  scenario: ScenarioDefinition;
  scenarioChunks: ScenarioChunkDeltaState;
  initialFrame: HumanoidWorldSnapshot;
  frameBuffer: HumanoidFrameBuffer;
  live: boolean;
  mode: HumanoidCameraMode;
  fitRevision: number;
  onReady: () => void;
  onFailure: () => void;
}

function StageScene(props: StageSceneProps): React.JSX.Element {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const bounds = props.scenario.bounds;
  const extent = Math.max(bounds.width, bounds.depth);
  return (
    <>
      <color attach="background" args={["#050505"]} />
      <fog attach="fog" args={["#050505", Math.max(18, extent * 0.9), Math.max(64, extent * 3.4)]} />
      <hemisphereLight args={[0xc7c7c7, 0x050505, 0.86]} />
      <ambientLight color={0xffffff} intensity={0.2} />
      <directionalLight
        castShadow
        color={0xffffff}
        intensity={2.35}
        position={[bounds.width * 0.45, 9.5, bounds.depth * 0.9]}
        shadow-mapSize-width={SHADOW_MAP_SIZE}
        shadow-mapSize-height={SHADOW_MAP_SIZE}
        shadow-bias={-0.0008}
        shadow-normalBias={0.02}
      />
      <directionalLight color={0xb8b8b8} intensity={0.56} position={[-bounds.width * 0.6, 4.5, -bounds.depth * 0.2]} />
      <directionalLight color={0xffffff} intensity={0.2} position={[bounds.width * 0.1, 3.2, -bounds.depth * 1.1]} />
      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping={false}
        minDistance={1.2}
        maxDistance={Math.max(40, extent * 3.5)}
        maxPolarAngle={Math.PI * 0.49}
      />
      <WorldRuntime {...props} controlsRef={controlsRef} />
    </>
  );
}

function WorldRuntime(
  props: StageSceneProps & { controlsRef: React.RefObject<OrbitControlsImpl | null> }
): null {
  const { camera, gl, invalidate, scene, size } = useThree();
  const worldRef = useRef<HumanoidWorldScene | null>(null);
  const startupRef = useRef({
    runId: props.runId,
    scenario: props.scenario,
    initialFrame: props.initialFrame
  });
  if (startupRef.current.runId !== props.runId) {
    startupRef.current = {
      runId: props.runId,
      scenario: props.scenario,
      initialFrame: props.initialFrame
    };
  }
  const scenarioChunksRef = useRef(props.scenarioChunks);
  scenarioChunksRef.current = props.scenarioChunks;
  const currentRef = useRef(props.frameBuffer.latest ?? props.initialFrame);
  const previousRootRef = useRef(new THREE.Vector3(
    currentRef.current.robot.rootPosition.x,
    currentRef.current.robot.rootPosition.y,
    currentRef.current.robot.rootPosition.z
  ));
  const fittedRef = useRef(false);
  const [worldRevision, setWorldRevision] = useState(0);
  const liveRef = useRef(props.live);
  const modeRef = useRef(props.mode);
  liveRef.current = props.live;
  modeRef.current = props.mode;

  useEffect(() => {
    const abort = new AbortController();
    let created: HumanoidWorldScene | null = null;
    const startup = startupRef.current;
    if (!props.frameBuffer.latest) props.frameBuffer.reset(startup.initialFrame);
    void HumanoidWorldScene.create(startup.scenario, scenarioChunksRef.current, abort.signal)
      .then((world) => {
        if (abort.signal.aborted) {
          world.dispose();
          return;
        }
        created = world;
        worldRef.current = world;
        const frame = props.frameBuffer.latest ?? startup.initialFrame;
        currentRef.current = frame;
        world.updateScenarioChunks(scenarioChunksRef.current);
        world.update(frame);
        scene.add(world.root);
        fittedRef.current = false;
        setWorldRevision((value) => value + 1);
        props.onReady();
        invalidate();
      })
      .catch(() => {
        if (!abort.signal.aborted) props.onFailure();
      });
    return () => {
      abort.abort();
      if (created) {
        scene.remove(created.root);
        created.dispose();
      }
      if (worldRef.current === created) worldRef.current = null;
    };
  }, [invalidate, props.frameBuffer, props.onFailure, props.onReady, props.runId, scene]);

  useEffect(() => props.frameBuffer.subscribe(invalidate), [invalidate, props.frameBuffer]);

  useEffect(() => {
    const world = worldRef.current;
    if (!world) return;
    world.updateScenarioChunks(props.scenarioChunks);
    world.update(currentRef.current);
    invalidate();
  }, [invalidate, props.scenarioChunks]);

  useEffect(() => {
    const world = worldRef.current;
    const controls = props.controlsRef.current;
    if (!world || !controls || !(camera instanceof THREE.PerspectiveCamera)) return;
    fitCamera(camera, controls, world, currentRef.current, props.scenario, props.mode, size);
    previousRootRef.current.copy(vector(currentRef.current.robot.rootPosition));
    fittedRef.current = true;
    invalidate();
  }, [camera, invalidate, props.controlsRef, props.fitRevision, props.mode, props.scenario, size.height, size.width, worldRevision]);

  useFrame((state) => {
    const world = worldRef.current;
    const controls = props.controlsRef.current;
    if (!world || !controls || !(camera instanceof THREE.PerspectiveCamera)) return;
    const visual = props.frameBuffer.sample(state.clock.elapsedTime * 1_000, liveRef.current)
      ?? props.frameBuffer.latest
      ?? currentRef.current;
    if (visual !== currentRef.current) {
      world.update(visual);
      currentRef.current = visual;
    }
    if (!fittedRef.current) {
      fitCamera(camera, controls, world, visual, props.scenario, modeRef.current, size);
      fittedRef.current = true;
    } else if (modeRef.current === "head") {
      placeHeadCamera(camera, visual);
    } else if (modeRef.current === "follow") {
      const root = vector(visual.robot.rootPosition);
      const delta = root.clone().sub(previousRootRef.current);
      camera.position.add(delta);
      controls.target.add(delta);
      controls.update();
      previousRootRef.current.copy(root);
    }
    publishRobotScreenBounds(gl.domElement, world.robotBounds(), camera, size, modeRef.current);
    if (liveRef.current && props.frameBuffer.pending) invalidate();
  });

  return null;
}

function fitCamera(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControlsImpl,
  world: HumanoidWorldScene,
  frame: HumanoidWorldSnapshot,
  scenario: ScenarioDefinition,
  mode: HumanoidCameraMode,
  size: { width: number; height: number }
): void {
  const extent = Math.max(scenario.bounds.width, scenario.bounds.depth);
  camera.up.set(0, 1, 0);
  camera.far = Math.max(140, extent * 5);
  camera.aspect = Math.max(1, size.width) / Math.max(1, size.height);
  camera.updateProjectionMatrix();
  if (mode === "head") {
    controls.enabled = false;
    placeHeadCamera(camera, frame);
    return;
  }
  controls.enabled = true;
  if (mode === "world") {
    const center = world.worldBounds().getCenter(new THREE.Vector3());
    controls.target.copy(center);
    camera.position.set(
      center.x + scenario.bounds.width * 0.62,
      Math.max(12, extent * 0.7),
      center.z + scenario.bounds.depth * 0.62
    );
    controls.update();
    return;
  }
  const robotBounds = world.robotBounds();
  const robotCenter = robotBounds.getCenter(new THREE.Vector3());
  const robotHeight = Math.max(1.25, robotBounds.getSize(new THREE.Vector3()).y);
  const rotation = quaternion(frame.robot.rootRotation);
  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(rotation);
  const side = new THREE.Vector3(1, 0, 0).applyQuaternion(rotation);
  const portrait = camera.aspect < 0.72;
  const distance = Math.max(portrait ? 5.25 : 3.05, robotHeight * (portrait ? 4.35 : 2.35));
  controls.target.copy(robotCenter).addScaledVector(forward, portrait ? 0.14 : 0.34);
  controls.target.y = robotCenter.y - robotHeight * (portrait ? 0 : 0.07);
  camera.position.copy(controls.target)
    .addScaledVector(forward, -distance)
    .addScaledVector(side, distance * (portrait ? 0.07 : 0.17));
  camera.position.y = controls.target.y + robotHeight * (portrait ? 0.84 : 0.76);
  controls.update();
}

function placeHeadCamera(camera: THREE.PerspectiveCamera, frame: HumanoidWorldSnapshot): void {
  const head = frame.robot.links.head_link;
  if (!head) return;
  const rotation = quaternion(head.rotation);
  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(rotation);
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(rotation);
  camera.position.copy(vector(head.position)).addScaledVector(up, 0.43).addScaledVector(forward, 0.08);
  camera.up.copy(up);
  camera.lookAt(camera.position.clone().addScaledVector(forward, 6));
}

function publishRobotScreenBounds(
  canvas: HTMLCanvasElement,
  bounds: THREE.Box3,
  camera: THREE.PerspectiveCamera,
  size: { width: number; height: number },
  mode: HumanoidCameraMode
): void {
  if (bounds.isEmpty()) return;
  const minimum = bounds.min;
  const maximum = bounds.max;
  const corners = [
    [minimum.x, minimum.y, minimum.z], [minimum.x, minimum.y, maximum.z],
    [minimum.x, maximum.y, minimum.z], [minimum.x, maximum.y, maximum.z],
    [maximum.x, minimum.y, minimum.z], [maximum.x, minimum.y, maximum.z],
    [maximum.x, maximum.y, minimum.z], [maximum.x, maximum.y, maximum.z]
  ] as const;
  let left = Number.POSITIVE_INFINITY;
  let right = Number.NEGATIVE_INFINITY;
  let top = Number.POSITIVE_INFINITY;
  let bottom = Number.NEGATIVE_INFINITY;
  for (const [x, y, z] of corners) {
    const projected = new THREE.Vector3(x, y, z).project(camera);
    left = Math.min(left, (projected.x + 1) * size.width / 2);
    right = Math.max(right, (projected.x + 1) * size.width / 2);
    top = Math.min(top, (1 - projected.y) * size.height / 2);
    bottom = Math.max(bottom, (1 - projected.y) * size.height / 2);
  }
  const revision = Number(canvas.dataset.robotProjectionRevision ?? "0") + 1;
  canvas.dataset.robotProjectionRevision = String(revision);
  canvas.dataset.robotScreenBounds = JSON.stringify({ left, right, top, bottom, mode, revision });
}

function vector(value: { x: number; y: number; z: number }): THREE.Vector3 {
  return new THREE.Vector3(value.x, value.y, value.z);
}

function quaternion(value: { x: number; y: number; z: number; w: number }): THREE.Quaternion {
  return new THREE.Quaternion(value.x, value.y, value.z, value.w);
}

function supportLabel(value: "double" | "left" | "right" | "none"): string {
  if (value === "double") return "双脚";
  if (value === "left") return "左脚";
  if (value === "right") return "右脚";
  return "腾空";
}

export const HumanoidStage = memo(HumanoidStageComponent);
