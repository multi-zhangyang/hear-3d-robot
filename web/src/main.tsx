// Bundled rather than fetched from a font CDN: the operator UI has to render
// identically offline and in CI, where no external request will resolve.
import "@fontsource-variable/space-grotesk";
import "@fontsource-variable/jetbrains-mono";
import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./styles.css";
import "./styles/activity.css";
import "./styles/agent-flow.css";
import "./styles/game-shell.css";
import "./styles/game-world.css";
import "./styles/journey.css";
import "./styles/mission-modal.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing application root");

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
