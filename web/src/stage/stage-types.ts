import type { CameraMode } from "./WorldHud";
import type { FirstPersonStatus } from "./stage-interaction";
import type { WorldSelection } from "./world-selection";

export interface StageCallbacks {
  onError: (message: string | null) => void;
  onSelection: (selection: WorldSelection | null) => void;
  onFirstPersonStatus: (status: FirstPersonStatus) => void;
}

export interface StageController {
  setLive: (live: boolean) => void;
  setCameraMode: (mode: CameraMode) => void;
  fit: () => void;
  clearSelection: () => void;
  dispose: () => void;
}
