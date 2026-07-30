import { describe, expect, it } from "vitest";
import { renderPixelRatio, SHADOW_MAP_SIZE } from "./render-quality";

describe("render quality", () => {
  it("caps dense mobile displays independently of device DPR", () => {
    expect(renderPixelRatio(390, 844, 3)).toBe(1.25);
  });

  it("bounds desktop fill rate and keeps a usable lower limit", () => {
    expect(renderPixelRatio(1440, 960, 2)).toBeGreaterThan(1.4);
    expect(renderPixelRatio(1440, 960, 2)).toBeLessThanOrEqual(1.5);
    expect(renderPixelRatio(3840, 2160, 2)).toBeGreaterThanOrEqual(0.75);
    expect(renderPixelRatio(3840, 2160, 2)).toBeLessThan(1);
  });

  it("uses a mobile-safe shadow map", () => {
    expect(SHADOW_MAP_SIZE).toBe(1024);
  });
});
