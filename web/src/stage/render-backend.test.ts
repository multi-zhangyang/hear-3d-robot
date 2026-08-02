import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { createRenderBackend, renderBackendKind } from "./render-backend";

describe("renderBackendKind", () => {
  it("reports the backend selected by Three.js after initialization", () => {
    expect(renderBackendKind({ isWebGPUBackend: true })).toBe("webgpu");
    expect(renderBackendKind({ isWebGLBackend: true })).toBe("webgl2");
    expect(() => renderBackendKind({})).toThrow("unknown rendering backend");
  });

  it("stops an obsolete stage before allocating a renderer", async () => {
    const initialization = new AbortController();
    initialization.abort();

    await expect(createRenderBackend(
      new THREE.Scene(),
      { onDeviceLost: vi.fn(), onError: vi.fn() },
      initialization.signal
    )).rejects.toMatchObject({ name: "AbortError" });
  });
});
