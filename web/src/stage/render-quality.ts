export const SHADOW_MAP_SIZE = 1024;

const MOBILE_WIDTH = 560;
const MOBILE_MAX_DPR = 1.25;
const DESKTOP_MAX_DPR = 1.5;
const MAX_RENDER_PIXELS = 2_800_000;
const MIN_DPR = 0.75;

/** Bounds fill-rate without assuming that a high device DPR means a fast GPU. */
export function renderPixelRatio(
  width: number,
  height: number,
  devicePixelRatio: number
): number {
  const safeWidth = Math.max(1, width);
  const safeHeight = Math.max(1, height);
  const device = Number.isFinite(devicePixelRatio) ? Math.max(MIN_DPR, devicePixelRatio) : 1;
  const deviceLimit = safeWidth <= MOBILE_WIDTH ? MOBILE_MAX_DPR : DESKTOP_MAX_DPR;
  const pixelLimit = Math.sqrt(MAX_RENDER_PIXELS / (safeWidth * safeHeight));
  return Math.max(MIN_DPR, Math.min(device, deviceLimit, pixelLimit));
}
