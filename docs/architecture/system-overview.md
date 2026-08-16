# HEAR System Architecture

Status: implementation overview for the neural-hierarchy V3 runtime.

HEAR is a hybrid system: model Agents choose semantic goals and Skills, the
Harness owns causality and authority, and learned/deterministic controllers
close the fast physical loop. Offline training improves the controller; it
does not move trajectory authority into the LLM hierarchy.

![HEAR hierarchical embodied agent architecture](hear-system-architecture.svg)

```mermaid
flowchart TB
    subgraph OFFLINE["Offline policy production · Colab GPU"]
      CONTRACT["Versioned task + observation contract"]
      TRAIN["mjlab / RSL-RL training"]
      GATE["Independent held-out physical gate"]
      POLICY["Qualified ONNX assets<br/>body + reach + contact"]
      CONTRACT --> TRAIN --> GATE --> POLICY
    end

    subgraph CORTEX["Model control · 14 isolated OpenAI Agents SDK Agents"]
      EXEC["Executive<br/>only root"]
      GOAL["Goal Valuation"]
      SELECT["Action Selection<br/>only Skill commitment owner"]
      PERCEPTION["Perception Manager<br/>parallel Scene + Memory join"]
      SCENE["Scene Interpreter<br/>current world semantics"]
      MEMORY["Relevant Memory<br/>historical outcomes"]
      SENSORIMOTOR["Sensorimotor Manager<br/>parallel Affordance + Risk join"]
      AFFORDANCE["Affordance<br/>bounded Skill candidates"]
      RISK["Risk / Interoception<br/>safety inhibition"]
      PREDICTIVE["Predictive Critic<br/>rollout comparison"]
      PREMOTOR["Premotor"]
      INTENT["Motor Intent<br/>lowest cognitive planning boundary"]
      DISPATCH["Certified Execution Dispatcher<br/>required tool · thinking disabled"]
      RECOVERY["Recovery<br/>exclusive bounded lease"]

      EXEC --> GOAL
      EXEC --> SELECT
      SELECT --> PERCEPTION
      SELECT --> SENSORIMOTOR
      PERCEPTION --> SCENE
      PERCEPTION --> MEMORY
      SENSORIMOTOR --> AFFORDANCE
      SENSORIMOTOR --> RISK
      SENSORIMOTOR --> PREDICTIVE
      SENSORIMOTOR --> PREMOTOR --> INTENT
      SENSORIMOTOR --> DISPATCH
      SENSORIMOTOR --> RECOVERY
    end

    subgraph HARNESS["Authority Harness · typed, causal, durable"]
      SIGNALS["World-versioned neural signals<br/>invocation + parent episode"]
      COMMIT["Goal epoch + Skill commitment"]
      SCHED["Event responsibility scheduler<br/>not an Agent"]
      CERT["One-use rollout certificate"]
      LEDGER["Atomic execution ledger"]
    end

    subgraph BODY["Embodied control · no LLM motor commands"]
      FUSION["Sensor Fusion<br/>deterministic"]
      ROLLOUT["MuJoCo Rollout Gate<br/>isolated state copies"]
      WRITER["Serial Executor<br/>sole physical writer"]
      CONTROL["50 Hz controller / reflex<br/>G1 body + reach + contact"]
      PLANT["200 Hz MuJoCo Body<br/>authoritative state"]
    end

    subgraph OPERATOR["Read-only projection and engineering surfaces"]
      API["Checkpoint + event stream"]
      PRODUCT["HEAR Operator<br/>R3F world + React Flow hierarchy"]
      FOX["Foxglove<br/>MCAP engineering telemetry"]
      RERUN["Rerun<br/>optional recording / replay"]
      API --> PRODUCT
      API -. "read-only MCAP" .-> FOX
      API -.-> RERUN
    end

    POLICY -. "installed only after gate" .-> CONTROL
    PERCEPTION --> SIGNALS
    SENSORIMOTOR --> SIGNALS
    SELECT --> COMMIT
    INTENT --> ROLLOUT
    ROLLOUT --> CERT
    CERT -. "one-use certificate input" .-> DISPATCH
    DISPATCH --> LEDGER --> WRITER
    WRITER --> CONTROL --> PLANT
    PLANT -. "body sensation" .-> FUSION
    FUSION -. "sensory evidence" .-> PERCEPTION
    ROLLOUT -. "rollout result" .-> PREDICTIVE
    PREDICTIVE -. "prediction verdict" .-> SENSORIMOTOR
    WRITER -. "physical receipt" .-> DISPATCH
    DISPATCH -. "typed completion" .-> SENSORIMOTOR
    WRITER -. "physical event" .-> SCHED
    SCHED -. "wake nearest responsible ancestor" .-> EXEC
    PLANT --> API
    SIGNALS --> API

    classDef agent fill:#173a31,stroke:#65e6bb,color:#effff8,stroke-width:1.5px
    classDef harness fill:#252f3c,stroke:#6ca9d1,color:#f1f8ff,stroke-width:1.5px
    classDef physical fill:#41331f,stroke:#d8ad67,color:#fff8e9,stroke-width:1.5px
    classDef writer fill:#214c3e,stroke:#9bf3d4,color:#ffffff,stroke-width:2.8px
    classDef offline fill:#302e3b,stroke:#9a95b5,color:#f6f3ff,stroke-width:1.3px
    classDef ui fill:#26352f,stroke:#7ca38f,color:#edf7f1,stroke-width:1.2px
    class EXEC,GOAL,SELECT,PERCEPTION,SCENE,MEMORY,SENSORIMOTOR,AFFORDANCE,RISK,PREDICTIVE,PREMOTOR,INTENT,DISPATCH,RECOVERY agent
    class SIGNALS,COMMIT,SCHED,CERT,LEDGER harness
    class FUSION,ROLLOUT,CONTROL,PLANT physical
    class WRITER writer
    class CONTRACT,TRAIN,GATE,POLICY offline
    class API,PRODUCT,FOX,RERUN ui
```

## How to read the diagram

- Solid arrows in the Agent region are direct control ownership. The complete
  19-node single-parent tree is defined in
  [neural-hierarchy-v3.md](neural-hierarchy-v3.md).
- Dotted arrows are evidence, feedback, offline deployment, or scheduler
  wake-up. They never create another control parent.
- Free cognition and planning end at Motor Intent. It emits bounded semantic
  motor intent, not joint values, trajectories, poses selected by Executive, or
  physical writes. The only lower model node is the non-thinking Certified
  Execution Dispatcher, whose required call can target only the serial writer.
- A rollout result must be accepted and converted to a one-use certificate
  before the atomic ledger can admit Serial Executor. Only that executor can
  mutate the authoritative MuJoCo body.
- The controller is a cascade: trained G1 body control remains active, the
  reach policy supplies the 14-dimensional upper-body reference, and the
  contact policy receives the 8-dimensional hand authority only during an
  admitted contact-rich Skill.
- The UI, Foxglove, and Rerun are projections. None is part of the control
  hierarchy and none may author a world revision or success receipt.

## Runtime cadences

| Band | Activation | Authority |
|---|---|---|
| Executive / Goal | mission and terminal events | goal continuation or retirement |
| Perception / Sensorimotor | world and Skill events | belief and bounded Skill proposal |
| Premotor / Motor Intent | committed Skill and rollout events | semantic motor intent |
| Dispatcher / Serial Executor | one admitted physical transaction | required model dispatch and sole write |
| Controller / MuJoCo | 50 Hz / 200 Hz continuous loop | actuator command / physical truth |
