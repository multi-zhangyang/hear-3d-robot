# HEAR Agent Harness v2

> Historical design only. This Coordinator-based runtime is retired and is
> not imported by the production mission runner. The authoritative design is
> [Neural Hierarchy V3](neural-hierarchy-v3.md); V2 remains solely to explain
> legacy run/manifest migration and earlier training decisions.

## Product thesis

HEAR is a hierarchical Agent Harness for an autonomous humanoid. The Harness
owns semantic Goals, typed Skill graphs, authority, evidence, recovery, and
audit. A robot policy owns closed-loop motion inside an authorized Skill
window. The Harness must not synthesize or supervise every control frame.

The system boundary is:

```text
Goal Manager / Coordinator
  -> typed Skill graph
  -> EmbodiedSkillCall
  -> learned Skill policy or behavior foundation model (10-50 Hz)
  -> whole-body controller and safety shield (200-500 Hz)
  -> sparse progress, success, failure, and environment-change events
  -> recovery, policy handoff, or replanning
```

Dense reference trajectories remain an offline teacher, a physical-preview
artifact, and a deterministic fallback. They are not the primary runtime
action language.

## Runtime roles

- **Goal Manager** keeps the Goal DAG and completion evidence. It runs only on
  goal, failure, or environment-change events.
- **Coordinator** is the principal reasoning Agent. It selects and composes
  typed Skills, then consumes physical receipts.
- **Grounding/Monitor service** replaces an LLM Sentry. It updates object,
  contact, articulation, safety, and contract evidence asynchronously.
- **Motion router** selects an embodied Skill policy and its compact command.
  It does not normally write dense trajectories.
- **Deterministic execution gate** replaces an LLM Executor. It validates
  hashes, world revision, contact authority, policy entry conditions, and
  safety limits before admitting a Skill Call.

The runtime now enforces this split directly. Only Goal Manager, Coordinator,
and Motion own model facades and persistent Sessions. The coordinator's
`delegate_humanoid_sentry` call executes `observe_humanoid` in-process, while
`delegate_physics_executor` maps an accepted planning receipt to exactly one
of `execute_humanoid_skill`, `execute_whole_body_motion`,
`execute_humanoid_navigation`, or `remove_world_block`. The outer Coordinator
tool call remains the durable model decision authority; the service identity
remains the action actor. Neither service can create a second model response
or rewrite the delegated action.

This boundary is enforced below the tool UI. A Sentry response cannot authorize
`observe_humanoid`, and an Executor response cannot authorize physical
execution or world mutation. These actions require a Coordinator tool call and
the matching deterministic contract (`grounding_monitor_v1` or
`execution_gate_v1`). For physical execution, the action-execution ledger
persists both the Coordinator decision reference and the complete Execution
Gate envelope before frame zero. A process restart revalidates and replays that
exact envelope; it cannot mint a replacement Executor decision, change the
accepted plan, or drop to a weaker recovery path. The referenced Coordinator
model lifecycle is retained independently of the bounded recent-history
window until the physical transaction is terminal.

Action chronology is explicit rather than inherited from JSON object key
order. The Harness assigns a monotonic `commitSequence` before staging each
durable receipt. Coordinator phase transitions, latest-plan selection,
post-execution Sentry barriers, world-mutation causality, block-removal
authority, context projection, and hot-window pruning all consume this order.
Historical V6 receipt windows are sequenced once at runtime restoration; mixed
or duplicate sequences fail checkpoint validation. Re-serializing or sorting
checkpoint keys therefore cannot change which plan or physical evidence is
authoritative.

The bounded `committed_actions` checkpoint window is also a cache, not an
authorization database. After pending outbox recovery, startup requires every
hot receipt to have one append-only `action_identities` tombstone and the exact
hash-bound row in both the `actions` and durable runtime-event journals. It
recomputes the action fingerprint from actor, typed action, and input; verifies
the event embeds that exact action row; and revalidates the receipt's completed
model lifecycle, manifest epoch, role, cycle, tool call, arguments, and
Coordinator delegation. A checkpoint-only receipt, missing journal row,
rebound event ID, or failed/pruned model lifecycle stops recovery before the
receipt can influence Coordinator phase or physical authority.

The same commit event anchors the post-action `action_runtime_state`. The
registered Skill DAG, selected strategy, active binding, planning-to-Skill
binding, recovery policy, transit-clearance obligation, and latest physical
revision are compared byte-for-byte with the state embedded in the latest
hash-bound action event. Their schemas remain useful for local consistency,
but a self-consistent checkpoint edit no longer creates durable Skill or
recovery authority.

Deterministic service delegation is durable as an envelope, not inferred from
the actor name after restart. Every new action commit event records the exact
Coordinator tool name, original source-input hash and value, deterministic
contract ID, and bound service-action input hash. Recovery reruns the same
`grounding_monitor_v1` or `execution_gate_v1` mapping used at live dispatch;
`ModelDecisionRef` alone cannot authorize a Sentry observation, physical
execution, or world mutation.

MuJoCo checkpoints are externally anchored as well. Whenever a persisted
physical cut changes, the runtime first appends a compact durable event with
the frame, world revision, and canonical SHA-256 of the full world checkpoint,
then writes that event identity into the checkpoint. Startup reconstructs the
MuJoCo cut, verifies the hash, and resolves the referenced append-only event.
An interrupted pre-checkpoint append is merely an orphan anchor; a modified or
rolled-back checkpoint cannot manufacture a matching journal event.

The Goal DAG follows the same append-before-checkpoint rule. Every changed DAG
is embedded in a content-addressed durable event after its model-call,
predicate-observability, dependency, and evidence validation succeeds. The
checkpoint stores that event ID and DAG state hash. Recovery resolves the
append-only event before trusting candidates, selection, retirement, or
completion, so recomputing the DAG's internal hashes inside a modified
checkpoint does not rewrite the hierarchical Agent's intent history.
The same anchor hashes the active cycle (including replan budget), physical
predicate progress, checker result, and cycle index. A self-consistent edit to
stable-frame counters or active-cycle identity therefore cannot make a Goal
finish early or grant a fresh planning budget after restart.

Embodied memory is likewise a derived cache rather than unverified planner
authority. Before a checkpoint can expose updated episodes, action experiences,
semantic indexes, or lifetime outcome counts, the runtime appends the complete
memory state and its canonical SHA-256 to a content-addressed durable event.
Recovery resolves that event and verifies the checkpoint state byte-for-byte.
This prevents a delayed checkpoint edit from poisoning later Skill selection
or recovery hints while preserving event-driven, cross-episode learning.

Compacted model working memory uses the same rule. The complete per-Agent
summary state, raw and retained history-chain identities, token calibration,
and compaction counters are content-addressed before checkpoint publication.
The resumable SDK RunState fingerprint also binds this memory hash. A modified
or downgraded context checkpoint is therefore discarded or rejected instead
of becoming a delayed instruction carrier after restart.

These persistence and Session recovery semantics define Harness contract epoch
15. A run created under an older Agent manifest must rotate its model Sessions
before it can resume under this authority kernel; deterministic service
identities and physical journals remain available for verified recovery.

Content identity alone is insufficient because an older valid checkpoint could
otherwise be replayed. Startup therefore requires each state anchor to equal
the latest append-only event of its type. The same monotonic head now covers
the complete physical execution ledger, including admission, committed frame
prefix, terminal identity, and acknowledgement. Removing an active intent can
no longer make an already-consumed plan appear executable again.

## Embodied Skill Call

Every physical Skill invocation carries:

- Skill, phase, Skill-plan node, binding transaction, invocation hash, and
  catalog hash;
- the observation frame and world revision from which authority was derived;
- an autonomous closed-loop execution window and event-driven replan policy;
- compact base twist, root height, wrist targets in pelvis coordinates,
  end-effector targets, and grasp requirements;
- a formal motion-option or navigation success contract;
- authorized contacts and mandatory stop conditions;
- requested progress, terminal, interruption, and environment-change events.

The low-level policy is free to choose joint motion inside this envelope. A
Skill is accepted only from observable physical evidence, never because a
model claims that it succeeded.

## Policy routing and handoff

Static capability labels are only the first admission filter. The target
routing record for each policy and Skill family contains:

- a calibrated success posterior with confidence bounds;
- an in-distribution / out-of-distribution score for the current physical
  state and command;
- a learned entry-state region and recent failure history;
- required observation features, embodiment, latency, and safety envelope;
- a transition policy that can guide the robot into the next policy's entry
  region before switching.

Handoffs blend commands only after both policies agree on morphology, timing,
and actuation. Transition states are deliberately sampled during training and
reported separately in evaluation.

HEAR implements the evidence-backed admission and handoff layer. Evidence is
partitioned by controller implementation and semantic Skill family. Cold-start
calls may explore; after enough real outcomes, a Wilson lower confidence bound
can deny an unreliable policy. Successful entry states and authorized commands
maintain online diagonal distributions used for state and command OOD checks.
When the learned policy is outside its successful entry region, Memory Bridge
uses reference control to track the successful 29-joint entry prototype with
zero root velocity. It releases only after five stable control steps satisfy
the state-OOD, joint-RMS, and joint-velocity gates, then uses the normal smooth
handoff. The bridge is bounded to four seconds, can abort on command drift or
posterior change, and is excluded from contact-rich or bimanual Skills where a
generic joint prototype could destroy task contact. Pending admissions,
bridge state, and learned evidence are checkpointed, while preview rollouts
are rolled back with their captured controller state.

Outcome attribution is segment-aware. Primary-policy, full-fallback, and
upper-body-overlay control steps are counted separately. A task that succeeds
only after full fallback intervention is not recorded as a learned-policy
success. Admission decisions, confidence bounds, OOD scores, fallback reasons,
and transition outcomes are carried by physical trajectories, terminal
receipts, and capability benchmarks.

## Training strategy

Training is auxiliary to the Harness and is conditioned on the exact command
shape used at deployment.

1. Train complementary teachers for locomotion, whole-body tracking,
   contact-rich manipulation, and recovery.
2. Collect planner-output-conditioned rollouts, including rejected plans,
   recovery states, policy entry states, and transition boundaries.
3. Distill teachers into a context-gated mixture-of-experts Skill policy using
   the `EmbodiedSkillCall` command and 10-20 control steps of history.
4. Balance sampling by Skill event and transition outcome, rather than by
   frame count. Failures and successful recoveries must not disappear inside
   long nominal trajectories.
5. Keep held-out seeds, object geometry, friction, mass, latency, controller
   perturbations, and planner command distributions separate from training.

The Workyard six-stage sequence may generate teacher data and full-task
evaluation episodes, but it must not be the deployment policy's hidden task
automaton. Deployment policies receive independent, typed Skill Calls.

The Workyard v2 actor observation therefore contains requested-capability
multi-hot values and autonomous-window progress instead of the teacher stage.
The stage machine remains private to curriculum command generation, reward
labels, and full-task evaluation. The deployable student is conditioned on the
same base, wrist, grasp, capability, and window fields used by
`HumanoidEmbodiedSkillCall v2`, and the contract requires success, failure, and
Harness-recovery trajectories to remain distinct training sources.

### Reach-teacher actuation boundary

The v8 reach preflight isolated a plant-tracking failure rather than an IK
failure. A target-conditioned feasible-posture map reached 97.58% of an
independent kinematic validation grid within 0.06 m, but its pure posture servo
still saturated both shoulder-roll command leads in every runtime environment
and never brought the measured wrists onto the target. Continuing to tune the
kinematic teacher against that plant would hide the actual fault.

The v9 environment therefore partitions actuation by the same ownership
boundary used by the Harness. The frozen locomotion plant retains the exact
mjlab source stiffness, damping, and armature for all twelve leg and three waist
joints. The fourteen residual-owned shoulder, elbow, and wrist joints use the
production task-tracking stiffness (`arm=80`, `wrist=40`), with damping scaled
by the square root of the stiffness ratio. Joint-level Unitree G1 effort limits
remain unchanged, hands and waist remain locked, and neither the learned policy
nor the analytic teacher receives authority outside the existing 14D residual
window. This is plant alignment, not an expansion of physical authority.

### Harness-in-the-loop data boundary

The durable training unit is one semantic Skill Call, not one sampled physics
frame and not one complete mission. Its manifest joins the exact Skill binding,
all planning attempts, the accepted execution, sparse Skill events, terminal
status, recovery lineage, controller routing, and the authoritative trajectory
identity. Planning rejection, physical failure, interruption, environment
change, recovery success, and recovery failure remain separate outcomes.

`PhysicalTrajectorySummary` is deliberately capped at 64 adaptive audit
samples. It is sufficient for evidence, routing attribution, benchmarking, and
replanning, but it must never be presented as a dense imitation-learning
rollout. Dense observation/action/teacher targets require a separate policy
frame sink keyed by Skill Call ID and an exact frame range. Checkpoints,
rollouts, and training products remain outside Git.

The exporter is available as:

```text
hear export-harness-rollouts --runs-dir PATH [--output FILE]
```

This separation lets HEAR use deployment experience for curriculum selection,
event-balanced sampling, teacher distillation, and failure analysis without
weakening the runtime audit log or fabricating missing policy data.

## Safety and verification

- Motion-option predicates are the executable semantic contract.
- MuJoCo state, contact evidence, grasp evidence, and world revision are the
  physical authority.
- Whole-body constraints are enforced below the learned policy. A future
  input-to-state-safe control-barrier-function filter should minimally modify
  unsafe commands before dynamic whole-body control.
- Formal counterexamples may improve reusable Skill contracts offline. They
  never grant new runtime authority or bypass the deterministic safety gate.

## Runtime targets

- At most two slow model decisions for a nominal Skill: select/parameterize,
  then accept the terminal receipt. Control execution uses zero model calls.
- Progress events are driven by physical completion, predicate evidence, or
  stable-step changes; no per-frame LLM polling.
- Every terminal execution receipt contains a typed Skill status, confidence,
  controller identity, failure code, and recovery class.
- Benchmark policy transitions, learned-policy frame ratio, Skill success,
  recovery success, falls, unauthorized contacts, support margin, foot slip,
  joint-limit margin, and requested effort.

## Research basis (verified 2026-08-10)

- [HANDOFF](https://arxiv.org/abs/2606.06493): compact task-space interface and
  multi-teacher, context-gated MoE whole-body controller.
- [RoboHarness](https://arxiv.org/abs/2607.18060): capability memory, online
  evidence, policy entry regions, and Memory Bridge handoffs.
- [Guava](https://arxiv.org/abs/2606.18363) and
  [AgentSpec](https://arxiv.org/abs/2606.14674): semantic actions, iterative
  embodied loops, typed composition, and scaffold compatibility.
- [Cortex](https://arxiv.org/abs/2607.05377) and
  [HiRoC](https://arxiv.org/abs/2608.05999): planner/executor alignment,
  planner-output-conditioned training, and event-balanced transitions.
- [VASO](https://arxiv.org/abs/2606.05395): formal Skill contracts and
  counterexample-guided offline evolution.
- [SafeWBC](https://arxiv.org/abs/2605.25546): ISSf-CBF safety filtering between
  kinematic and dynamic whole-body control.
- [MotionPyramid](https://arxiv.org/abs/2606.20705),
  [Scaling BFM](https://arxiv.org/abs/2607.15163), and
  [OMG](https://arxiv.org/abs/2606.10340): reusable hierarchical motion
  representations and scalable humanoid behavior models.
- [RHO](https://arxiv.org/abs/2606.16458): optimize an interpretable harness
  repository offline to reduce deployment-time code/model loops.
- [EMS](https://arxiv.org/abs/2608.06434),
  [RoboBRIDGE](https://arxiv.org/abs/2607.27881), and
  [RPG](https://arxiv.org/abs/2604.21355): sparse slow-system use, asynchronous
  monitoring/recovery, and robust randomized policy transitions.

These works support the architecture; none is copied as a complete system.
HEAR's differentiator is the authority- and evidence-preserving integration of
hierarchical Agents with autonomous humanoid Skill policies.

## Frontier review and adoption order (verified 2026-08-11)

The following newer results sharpen the implementation order without changing
the product thesis:

- [Mimir](https://arxiv.org/abs/2608.04933) separates world memory from task
  memory and grounds the active goal against current evidence before every
  action. HEAR should implement this as a deterministic grounding receipt over
  the Goal DAG, object evidence, hand state, and active Skill binding, not as a
  fourth reasoning Agent.
- [HALO](https://arxiv.org/abs/2607.27636) rechecks localized obligations at
  dispatch time and preserves still-valid components of a heterogeneous Agent
  response. This matches HEAR's exact-action gate and suggests making each Skill
  dependency, contact authority, and evidence reference an independently
  revalidated obligation instead of invalidating an entire plan wholesale.
- [BRACE](https://arxiv.org/abs/2608.01428) treats replanning as a budgeted
  control loop with token and tail-latency SLOs. HEAR should attach a replan
  budget and deadline to every autonomous Skill window, then choose local
  recovery, policy switch, compact replan, or full Goal re-evaluation from
  physical failure class and remaining budget.
- [CheckVLA](https://arxiv.org/abs/2607.26789) uses an action-conditioned world
  model and conformal risk thresholds to verify open-loop chunks. Its useful
  lesson for HEAR is an optional event-triggered execution verifier below the
  Harness, calibrated to bound unnecessary interventions; it must not replace
  MuJoCo authority or poll an LLM per frame.
- [HERO](https://arxiv.org/abs/2607.26809),
  [CLIFT](https://arxiv.org/abs/2607.29172), and
  [LabEvolver](https://arxiv.org/abs/2607.27690) all turn closed-loop deployment
  experience into reusable skills or improved policies. HEAR's corresponding
  flywheel is durable Skill Call manifest -> dense teacher rollout ->
  event-balanced training -> held-out capability evidence -> guarded routing.
- [HumanCLAW](https://arxiv.org/abs/2607.27180) demonstrates why action
  intelligence and motor execution must be scored separately. HEAR benchmarks
  should report both semantic Skill selection quality and controller execution
  quality, with failure attribution never collapsed into one success number.
- [Claim Provenance and Verification Gate](https://arxiv.org/abs/2608.06830)
  shows that hierarchical communication itself is an attack surface. Goal,
  object, contact, and completion claims passed between HEAR Agents should
  remain evidence-addressed and be rechecked before downstream reuse.

Immediate implementation order:

1. Complete the Harness-in-the-loop Skill Call dataset and a separate dense
   policy-frame sink.
2. Replace high-variance whole-body PPO cold starts with a stable locomotion /
   balance teacher, upper-body task residuals, staged unfreezing, and then
   HANDOFF-style complementary-teacher distillation.
3. Add Mimir-style pre-action grounding receipts and HALO-style localized
   dispatch obligations.
4. Add BRACE-style replan budgets and dual semantic/controller attribution to
   the benchmark.
5. Add an ISSf-CBF safety filter; evaluate an action-conditioned verifier only
   after enough failure and perturbation trajectories exist to calibrate it.

Deferred research includes [RoboTTT](https://arxiv.org/abs/2607.15275),
[omega-0](https://arxiv.org/abs/2608.06375), and
[RoboReact](https://arxiv.org/abs/2608.03387). Long fast-weight context,
whole-body world-action models, and generated-video teachers are promising,
but HEAR currently lacks the dense, diverse, physically grounded dataset needed
to adopt them responsibly. They are future teacher or policy backends, not
replacements for the Harness.

### Additional screening on 2026-08-11

- [SkillMemo](https://arxiv.org/abs/2608.05970) retrieves compact latent Skill
  priors learned from expert-guided trajectory segmentation and improves both
  diffusion-policy and VLA backbones on compositional manipulation. HEAR may
  evaluate this inside a policy expert after the dense Skill-Call dataset is
  available. A retrieved latent segment is never a Harness Skill, cannot grant
  contact authority, and remains bounded by the typed Skill Call and physical
  success contract.
- [PhyAI](https://arxiv.org/abs/2608.03682) unifies onboard, edge, evaluation,
  and cloud-rollout inference through model adapters and a control-time
  roofline. It is relevant once HEAR carries several behavior foundation model
  families: the same checkpoint and action semantics should run in training and
  deployment. It is an inference substrate, not a planner, authority layer, or
  reason to move the control loop into the cloud.
- Distribution-aware reincarnating RL
  ([Scientific Reports](https://doi.org/10.1038/s41598-026-63862-9)) identifies
  two separate teacher-to-student shifts: offline OOD action extrapolation and
  the online replay transition from teacher to student experience. Its
  conservative value reuse and balanced replay idea is worth an ablation after
  teacher distillation. Current evidence is Atari rather than humanoid control,
  so it is not adopted as a training recipe without robot-policy validation.

The screening also found VLM planner-and-feedback systems that keep a language
model in the physical feedback loop. HEAR does not adopt that pattern: semantic
replanning stays event-triggered, while balance, locomotion, manipulation, and
safety remain autonomous low-level loops.

### Latest harness and policy screening on 2026-08-11

The final arXiv release window available on this date adds several concrete
design constraints:

- [A2E](https://arxiv.org/abs/2608.07346) evaluates the model-harness pair with
  standardized traces and separate planning, tool-use, recovery, correctness,
  and efficiency metrics. [Skill-Use](https://arxiv.org/abs/2608.04828) further
  separates Skill trigger, procedural compliance, and forbidden-boundary
  behavior. HEAR should adopt both decompositions in its benchmark: semantic
  Skill selection, contract compliance, authority-boundary compliance, and
  controller execution must remain separate measurements.
- [HarnessSafe](https://arxiv.org/abs/2608.06984) shows that persistent memory,
  Skills, tools, and shared artifacts are delayed attack carriers. HEAR's
  evidence-addressed Goal and Skill records need lifecycle tests that inject a
  tainted claim into each carrier, restart the run, and verify that dispatch
  grounding and physical authority still stop it.
- [CapLease](https://arxiv.org/abs/2608.01710) identifies semantic replay across
  retries and crash recovery as the real authorization problem. This validates
  HEAR's durable action-execution ledger, canonical action hashes, single
  admission, and idempotent physical sink; future authorization changes must
  preserve monotonic consumption rather than merely issuing fresh tokens.
- [Representation Handoffs](https://arxiv.org/abs/2608.07154) reports that
  registered Skill calls, grounded object poses, runtime bindings, and dry-run
  traces are useful integration boundaries. This independently supports HEAR's
  typed `EmbodiedSkillCall` and dispatch-grounding receipt instead of free-form
  language handoffs.
- [BCP](https://arxiv.org/abs/2608.03483) learns a continue-or-replan head for
  adaptive action-chunk horizons, while
  [ChainVLA](https://arxiv.org/abs/2608.02326) carries a revisable working state,
  sparse event memory, and the unexecuted motion tail across policy queries.
  These belong inside a Skill policy window. HEAR should eventually replace a
  fixed chunk length with a policy continuation head, but the head may only
  shorten or refresh an authorized Skill Call; it cannot extend contact,
  duration, or semantic authority.
- [AutoIntervene](https://arxiv.org/abs/2608.07065) calibrates asymmetric
  policy-to-recovery and recovery-to-policy thresholds from held-out successful
  rollouts. HEAR should use this for learned-policy/reference-policy handoffs,
  replacing hand-tuned OOD cutoffs only after the dense Skill dataset contains
  enough transition examples.
- [TEMPO](https://arxiv.org/abs/2608.07314) updates semantic projection slowly
  and the action expert quickly during RL post-training. This is a good fit for
  HEAR's staged training: preserve the Skill semantics and command interface,
  first adapt the low-level expert, then cautiously unfreeze the semantic gate.
- [AtlasVLA](https://arxiv.org/abs/2608.06729) and
  [PSG-JEPA](https://arxiv.org/abs/2608.06799) support persistent world/ego state
  and physically identifiable policy latents. Their memories can improve
  perception and policy state, but all claims crossing into the Harness still
  require evidence-addressed grounding.

These results do not justify replacing the Harness with one end-to-end VLA.
They sharpen the boundary: the Harness owns durable semantics and authority;
the learned policy owns adaptive action horizons and autonomous physical
feedback inside that authority.

### Source-level adoption decisions on 2026-08-11

A source-level review of the most relevant papers and released repositories
adds five concrete decisions. Evidence from robot hardware is kept separate
from evidence on software-agent benchmarks.

1. **Keep the hierarchy and strengthen its interfaces.**
   [What Matters in Orchestrating Robot Policies](https://arxiv.org/abs/2606.10267)
   unifies hierarchical VLA systems as options and finds that the planner,
   steerable controller, termination condition, observation representation,
   and memory mechanism jointly determine performance. Accurate success-based
   termination beats fixed or model-predicted horizons; grounded object/contact
   descriptions help more than raw images alone; raw in-episode history has
   little value, while cross-episode affordance summaries help. These findings
   support HEAR's deterministic physical checker, evidence-addressed current
   observation, typed Skill Call, and cross-episode outcome memory. They argue
   against adding more chat history or another per-step reasoning Agent.

2. **Make Harness evolution an offline, constrained optimization loop.**
   [Harness-R1](https://arxiv.org/abs/2608.02276),
   [HarnessCompass](https://arxiv.org/abs/2608.01918), and
   [Living-Harness](https://arxiv.org/abs/2607.26598) show that failure
   trajectories can improve context construction, memory retrieval, action
   mediation, and recovery. Their evidence is from software and interactive
   benchmarks, not physical robots, so HEAR must not enable runtime
   self-modification. The admissible design is: immutable authority kernel
   (schemas, action ledger, dispatch obligations, physical safety, and audit),
   editable soft shell (instructions, observation projection, retrieval,
   recovery hints, and Skill documentation), deterministic replay on the same
   failures, held-out worlds and model backends, regression and attack tests,
   then an explicit signed promotion. The existing semantic Skill manifests,
   dense policy frames, exact seeds, and failure attribution provide the data
   plane needed for this later loop.

3. **Calibrate continuation and handoff inside the authorized Skill window.**
   [BCP](https://arxiv.org/abs/2608.03483) and
   [ChainVLA](https://arxiv.org/abs/2608.02326) support a small policy-side
   continue/refresh/terminate decision and a revisable execution state instead
   of a fixed chunk timer. [AutoIntervene](https://arxiv.org/abs/2608.07065)
   supports separate, held-out calibrated thresholds for policy-to-recovery and
   recovery-to-policy transitions. HEAR may adopt these only below the Harness:
   continuation can shorten or refresh a current window but cannot extend its
   duration, contacts, object binding, or semantic authority. Current diagonal
   state OOD and hand-tuned thresholds remain a cold-start baseline; calibration
   requires enough successful and failed transition frames first.

4. **Use a deployable balance state before granting upper-body policies more
   whole-body authority.**
   [First Deployable Dynamic-CoM](https://arxiv.org/abs/2608.00500) provides a
   released G1 training and sim2sim stack and reports that support-relative
   planar CoM position and velocity are reconstructible from encoders, IMU, and
   the kinematic/mass model because absolute base velocity cancels. Its clean
   single-leg criterion also separates balance prevention from stepping or
   hopping recovery. HEAR should add this 4-D actor state (and derived capture
   point), support margin, time-to-boundary, foot displacement, contact loss,
   and jerk metrics before unfreezing waist authority. Code and policy are
   Apache-2.0, but the released motion data is GPL-3.0 and must not be copied
   into HEAR without a separate licensing decision.

5. **Replace cold reach exploration with a task-space teacher, then distill.**
   [CEER](https://arxiv.org/abs/2605.19981) and
   [HANDOFF](https://arxiv.org/abs/2606.06493) independently validate compact
   root/end-effector or base/height/wrist command interfaces. HANDOFF obtains a
   single policy through complementary locomotion, whole-body, and recovery
   teachers, action-sliced KL, DAgger, and context-gated MoE training. Its
   baseline adapter also gives a practical reach teacher: pelvis-relative
   differential IK with joint limits, damping, a bent-elbow ready pose, and a
   persistent warm start. Workyard v4 therefore uses a real 14-D arm/wrist
   policy with hands held at a stable open pose, a signed wrist-progress term,
   and online DAgger from a batched task-space IK teacher before PPO
   fine-tuning. The eight
   hand-synergy dimensions should enter only with contact/grasp objectives; a
   later 25-D waist-residual policy remains gated on reach and Dynamic-CoM
   acceptance.

The resulting adoption order is deliberately asymmetric. The immediate robot
work is 14-D teacher-guided reach plus Dynamic-CoM observation and evaluation.
The immediate Harness work is trace-level model--Harness benchmarking and
persistent-carrier attack replay. Offline Harness evolution follows only after
those evaluators can reject regressions. World-action models, generated-video
teachers, context-aware motion priors, and fast structured MoE training remain
candidate backends after HEAR has a sufficiently diverse Skill-conditioned
dense dataset; they do not change the authority architecture.

## First-class recovery budget implemented

Every active autonomous Cycle now owns a durable recovery budget. Deterministic
controller recovery and capability-router policy switching consume zero model
budget. A rejected semantic plan opens a bounded recovery window with:

- three compact Coordinator replan decisions;
- up to three Motion specialist calls per compact decision;
- one bounded Goal re-evaluation escalation with at most three Goal Manager
  calls;
- a 120-second recovery deadline and a 30-second per-call latency SLO;
- per-call started/completed/failed state, exact latency, and SLO-violation
  evidence in the checkpoint and runtime event stream.

`CURRENT HARNESS AUTHORITY.recovery_authority` exposes used and remaining
budget, deadline state, allowed recovery layers, in-flight calls, failures, and
tail-latency violations. Starting additional Coordinator or specialist model
calls after their tier is exhausted is rejected before the model request.
Internal navigation obstruction recovery remains a zero-model local tier with
its own two-attempt budget; terminal navigation receipts now report its failure
class, use, remaining budget, and explicit `model_calls_consumed: 0`.

## Dense policy data plane implemented

Authoritative semantic Skill execution now emits one exact record per control
step to `HEAR_DENSE_POLICY_ROLLOUT_DIR`. Motion execution, navigation, and
accepted online navigation replans carry the sink; preview simulations,
station keeping, and planning rollouts do not. Each per-call JSONL file contains
the exact controller observation/action tensors, pre/post physical state,
reference, final PD command, controller route, and supervision class.

The writer synchronizes every appended frame and maintains a forward SHA-256
chain. Recovery validates the complete chain, trims only a torn final JSONL
fragment, accepts deterministic replay of already-synced frames, rejects
divergent replay, and records explicit step/frame gaps instead of inventing
data. `export-harness-rollouts` verifies these files and attaches their hashes,
frame coverage, teacher counts, tensor protocols, and completeness evidence to
the durable semantic Skill manifest.

## Frozen-teacher residual Workyard v4

The failed 37D whole-body PPO cold start remains an explicit baseline, not the
main training recipe. Workyard v4 composes a frozen G1 velocity teacher, a
supervision-only analytic reach teacher, and a phase-one residual student:

- the teacher consumes its original 99D observation and produces the original
  29D action as one dynamic CUDA TorchScript batch, but its execution authority
  is sliced to lower body and waist;
- the 12 lower-body and three waist joint commands are exactly the teacher
  reference and cannot be modified by the phase-one actor;
- the student produces only 14 arm/wrist position residuals around the neutral
  upper-body pose; both hands remain at a fixed open target;
- a batched MuJoCo-Warp DLS teacher labels pelvis-relative wrist targets for
  online DAgger, but is absent from actor observation and owns no execution or
  Harness authority;
- the 231D student observation includes the current locomotion-teacher action,
  4D support-relative Dynamic-CoM, and the exact
  Skill task-space command but excludes teacher stage, target stage, curriculum
  promotion state, and the full task automaton;
- teacher frames, upper-body residual frames, composition error, frozen-joint
  tracking error, signed wrist progress, support margin, capture point, foot
  displacement/slip, effort, fall, and joint velocity are attributed separately.

The teacher exporter validates batched TorchScript output against the native
RSL-RL actor before publishing its hash. The Colab smoke refuses CPU round trips
inside the control loop, nonzero teacher gradient parameters, any composition
error, moving hand targets, or missing environment closure. A teacher-only mode
tests analytic guidance before spending a training run. Training then updates
the deployable actor with online Smooth-L1 DAgger and hands the same actor to
PPO. Smoke and warm-start runs do not grant deployment acceptance. Held-out
reach plus Dynamic-CoM acceptance may expand 14D to a 22D contact/grasp policy;
only that later phase can gate a separate 25D low-amplitude waist residual.

## Contact-policy safety governor

The contact actor owns only an authorized active hand's 8D coordination
increment. Its output is not sent directly to the 14 hand position servos. A
deterministic Harness governor resolves coordination to joint targets and caps
each closing target at 0.25 rad beyond the measured joint position. Opening is
never rate-limited by this guard, so the 6 N directional reflex and 12 N
emergency release can unload stored servo energy immediately. The same governor
runs in MJLab training and TypeScript MuJoCo deployment, and the value is bound
into the training contract, formal report, export report, controller command,
checkpoint serialization, and policy installer.

This guard fixes two different failure modes without giving the language-model
layer motor authority. First, a blocked fingertip cannot accumulate a large PD
spring error while the policy continues to request closure. Second, an
incidental opposing-finger contact at zero coordination no longer freezes an
unstable point contact: the analytic teacher establishes a shallow 0.40 support
curl while closing only the missing thumb. Force and opposition remain measured
MuJoCo facts, and the hierarchical coordinator still acts only through typed
Skills and physical acceptance receipts.

## First executable DAgger-to-PPO evidence

The first end-to-end L4 run on 2026-08-11 validates the training boundary but
does not yet validate a deployable reach Skill. With 64 environments, a
200-control-step teacher preflight, 100 online DAgger steps, two PPO iterations,
and a separate 300-step evaluation:

- the task-space teacher reduced mean wrist error from 0.208 m to a best mean
  of 0.113 m and reached 0.082 m in the best environment without a fall;
- DAgger trained the actual deployable actor on 6,400 visited-state labels,
  updated its observation normalizer with all 6,400 samples, and reduced
  Smooth-L1 loss from 0.05290 to 0.00417 (92.1%), with 100% label coverage;
- the same actor continued directly into PPO, producing readable checkpoints
  and scalar curves at 886--999 environment steps/s;
- the held-out-sized gate was deliberately not run, and the short evaluation
  achieved only 4.7% reach success. It had zero falls and zero illegal ground
  contacts, but it is not evidence for expanding hand or waist authority.

The run also exposed an evaluation bug rather than a policy success: contact
evidence was absent for the first three to four reset frames in every
environment. Dynamic-CoM evaluation now preserves reset-inclusive evidence,
records the first 16 control steps, and separately gates on a fixed post-settle
window whose foot-displacement origin is captured after settling. Thresholds
remain unchanged. A subsequent executable check reduced post-settle
no-foot-contact rate to zero, while support margin and foot-slip speed still
failed. Those remaining failures are physical policy/controller problems and
must not be tuned away merely to pass acceptance.

The immediate optimization target is therefore longer 14D reach distillation
and PPO retention, not a larger action space. Training now compares the DAgger
warm checkpoint against the PPO checkpoint on identical seeds and reports
action-clipping frequency. In the first deliberately tiny 16-environment
comparison, one PPO iteration reduced action clipping but lost 6.25 percentage
points of reach success, increased best-episode mean wrist error by 8.9 mm,
reduced minimum support margin by 3.8 cm, and increased peak foot slip by
0.114 m/s. This sample is too small for a policy-quality claim, but it proves
that PPO retention must be an explicit gate. PPO is useful only if it improves
task success and balance without erasing teacher-guided reach behavior.

## Rollout-wide Retention PPO evidence

HEAR now owns a project-local `HearRetentionPPO` extension instead of patching
MJLab or RSL-RL. Before every learner action, the current MuJoCo state receives
an analytic 14D reach-teacher label. The label is stored as a private rollout
field, remains absent from actor and critic observation groups, and follows the
exact RSL-RL shuffle into the same PPO minibatch. The joint objective combines
PPO with teacher Smooth-L1 on every learner-visited state. Five initial updates
are critic-only, the actor normalizer is frozen after DAgger, and actor/critic
learning rates remain separate.

RSL-RL 5.4's native Beta distribution structurally bounds both sampled and
deterministic actions to `[-1, 1]`. The first 64-environment, 20-update L4
ablation verified the integration and 40,960/40,960 rollout labels, but exposed
that constraining only the Beta mean left an average action standard deviation
near 0.39. Clipping fell to zero, yet PPO worsened minimum wrist error by
29.1 mm and peak foot slip by 0.094 m/s. The checkpoint selector correctly
rolled back to DAgger.

The second identical-scale run added a distribution-level constraint: teacher
mean Smooth-L1 plus a penalty above action standard deviation 0.15, teacher
coefficient 1.0, and no entropy bonus during this retention phase. Final mean
action standard deviation fell to 0.145 and clipping remained zero. Relative
to the identical-seed DAgger checkpoint, PPO then:

- preserved the 1.56% short-run success rate;
- improved minimum mean wrist error by 10.0 mm;
- improved minimum support margin by 3.94 mm;
- reduced peak foot slip by 0.146 m/s.

The retention selector therefore chose the PPO checkpoint in that run. This is
evidence that rollout-wide, distribution-aware retention works; it is not a
phase-one policy-quality acceptance. Absolute success is still only 1.56%, the
500-episode held-out requirement was not run, and support/foot-motion limits
still fail. Hand and waist checkpoint expansion remain unauthorized.

Reading Beta variance during DAgger initially consumed the actor's sampling RNG
and confounded the DAgger state distribution. The current code computes mean
and variance without sampling; that correction must be included in the next
longer 14D run.

## Long-pilot ceiling and grounded pre-grasp correction

The subsequent 256-environment pilot ran 1,000 DAgger control steps and 100 PPO
updates, for 1,075,200 environment steps, followed by a 256-by-600 identical-
seed comparison. DAgger reached 1.95% success and a 0.1120 m minimum mean wrist
error. PPO reached 2.34% and 0.1111 m, improved minimum support margin by
18.89 mm, but worsened final wrist error by 26.0 mm and peak foot slip by
0.0892 m/s. The retention gate rejected PPO and selected the DAgger checkpoint.

More training was not the next rational action. The analytic teacher-only
preflight itself reached only 2.34% success and a 0.1128 m minimum mean wrist
error, essentially the same ceiling as both learned checkpoints. A CPU MuJoCo
kinematic audit then found the underlying contract error: with the residual
authority fixed at +/-0.5 rad, none of an 18-point left/right/full-object-jitter
grid could satisfy the reach termination while targeting the rod's contact-side
point. The shoulder is roughly 0.41 m above the rod and the hand-locked reach
phase was being asked to extend the wrist almost onto a contact surface. That
target belonged to the later contact/grasp phase.

The corrected reach grounding keeps hand and waist authority locked and does
not widen the 14D action. The Harness entry stance moves from x=0.60 m to
x=0.63 m, and the reach target becomes the point 0.10 m from the rod along the
rod-to-corresponding-shoulder ray. It is a collision-avoiding pre-grasp command;
exact contact-side targets activate only after contact authority exists. The
reach transition now measures active-wrist error to the typed wrist command
against that command's tolerance. Rod distance remains contact/grasp evidence,
not a substitute completion condition. On the same MuJoCo grid, this raises the
within-0.06 m command-tracking kinematic ceiling from 0/18 to 18/18 while
retaining the original +/-0.5 rad arm/wrist authority. Even the stricter legacy
rod-distance check improves to 16/18 (88.9%). Dynamic teacher-only validation
remains the gate before another DAgger or PPO run.

Evaluation now also freezes success, terminal wrist error, and minimum wrist
error after each environment's first termination. RSL-RL immediately reuses a
terminated slot; evidence from that slot's next episode must not overwrite or
inflate the held-out first-episode result.

The first dynamic pre-grasp rerun isolated the remaining controller defect. On
an L4 with 128 environments for 200 control steps, the stateless v1 DLS teacher
had zero successful episodes, reduced mean command error only to 0.0971 m, and
then rebounded to 0.3399 m. Its arm authority was saturated on 15.5% of active
joint samples, dominated by left/right shoulder roll at 79.2%/90.6%. Geometry
was no longer the ceiling; converting a one-frame Cartesian delta directly into
a high-lead joint command caused overshoot and loss of the target.

The v2 reach teacher therefore added per-environment joint-target memory,
control-step idempotence, adaptive task-space DLS near singularities, null-space
posture motion, bounded target slew/lead, and hold hysteresis. Its first executable
128-by-200 run improved minimum mean error to 0.0844 m, but still had zero
successes and rebounded to 0.8174 m. Lead saturation occurred on 45.2% of active
joint samples and hold occupancy remained zero. The target memory was integrating
fresh DLS corrections while the measured joints lagged, producing controller
wind-up rather than the intended filter.

The v3 correction anchors every instantaneous IK target to the measured joint
position. Persistent state now only slew-limits movement toward that target and
is re-clamped to joint, authority, and command-lead boundaries every control
step. The DLS correction may request up to 0.12 rad, but solver-filter slew is
limited to 0.015 rad per control step and physical command lead to 0.08 rad.
Reports distinguish that bounded solver slew from any larger final target
recentring required when measured joint motion crosses a hard lead boundary. A 5/7.5 cm
enter/release hysteresis holds a reached pre-grasp. Reports retain authority,
soft-limit and lead saturation, actual target slew, active-arm minimum singular
value, and hold occupancy. The 85% success / 0.06 m minimum-mean-error preflight
remains mandatory before any new DAgger or PPO run.

The first local v3 screen ran 16 environments for 200 steps on the RTX 3050 Ti.
It removed the unstable tail: final rebound fell to 0.0124 m, post-settle peak
foot slip to 0.0183 m/s, and foot displacement to 0.0142 m, with no falls. But
minimum mean wrist error was still 0.1597 m and success remained zero. The safe
servo was underpowered, not accepted. V4 retains the measured-joint anchor and
anti-windup boundaries while increasing the bounded correction/lead/slew from
0.12/0.08/0.015 rad to 0.20/0.16/0.03 rad. Local teacher screening must show
useful command tracking without restoring v2's rebound or balance failure before
the 128-environment Colab gate is repeated.

That v4 screen improved the best part of the trajectory but exposed the next
structural error. Across 16 environments and 200 control steps, minimum mean
wrist error fell to 0.1005 m, yet success remained zero and final error rebounded
to 0.1790 m. Left shoulder yaw still saturated its authority on 59.4% of active
samples, right shoulder yaw on 33.9%, and left shoulder roll hit its lead bound
on 78.3%. Increasing command bandwidth again would hide, not solve, the 7-DoF
redundancy error: the DLS null space was pulling toward the standing neutral pose
while feasible reaches require shoulder roll, shoulder yaw, and elbow flexion.

V5 therefore preserves v4's measured-joint anchor, anti-windup, 0.20 rad joint
correction, 0.16 rad command lead, and 0.03 rad solver slew. Its only controller
change is a 0.20-gain null-space reference named
`offline-feasible-shoulder-ray-pregrasp-median-v1`. The reference is the
per-arm median of an 18-point MuJoCo feasibility grid at the production 0.5 rad
authority. That grid reaches the typed 0.06 m command tolerance in 18/18 cases,
with 0.0124 m mean and 0.0431 m maximum approach error. The normalized posture
is frozen in the training contract and converted through the configured action
scale, then clamped again to soft and Harness authority limits at runtime. This
makes the redundancy prior reviewable without granting any new action path to
the hands, waist, or frozen locomotion joints.

Teacher reports now include a sparse first-episode wrist-error trace at step 0,
every ten control steps, and the final step. Each point records only active
environment count and mean/p50/p90 error; it does not copy per-environment CUDA
vectors into the report. The local v5 screen remains mandatory, and Colab,
DAgger, and PPO remain disabled until the existing 85% success and 0.06 m
minimum-mean teacher gate is actually satisfied.

The local v5 screen confirmed that the fixed feasible posture was safer but not
sufficient. Authority saturation fell from v4's 8.56% to 1.78%, final mean error
improved from 0.1790 m to 0.1399 m, and post-settle foot slip stayed at
0.0289 m/s. However, minimum mean error remained 0.1026 m with zero successes;
left/right shoulder-roll lead saturation was still 92.3%/84.4%. A fixed median
posture covered only 322/578 (55.7%) points on a dense grid spanning the command
generator's real +/-0.08 m object jitter. The earlier 18/18 feasibility statement
covered only a narrower +/-0.05 m diagnostic grid and could not justify the full
training distribution.

V6 replaces that fixed null-space reference with a typed-target-conditioned
quadratic posture map. It was fitted per arm on 25 offline IK solutions covering
the full +/-0.08 m command range, using normalized target-pelvis x/y features
`[1, x, y, x^2, xy, y^2]`. On a separate 17-by-17 dense grid per arm, the frozen
map reaches 564/578 targets (97.58%) within 0.06 m, with 0.0202 m mean and
0.0469 m p90 error. The two far diagonal corners remain authority-limited at
about 0.079 m maximum error, which is consistent with the 85% teacher gate rather
than hidden by wider authority.

At runtime the quadratic map executes as batched CUDA tensor algebra and supplies
a task-compatible joint-space attractor. Adaptive DLS remains the feedback term
for dynamic base motion and actuator tracking. Feature inputs are bounded before
the polynomial, normalized actions are bounded before conversion to physical
joint targets, and the existing measured-joint anchor, soft limits, Harness
authority, lead limit, slew filter, anti-windup and hold hysteresis remain in
force. Reports count feature and normalized-action clamps so extrapolation cannot
silently masquerade as teacher quality. V6 still carries supervision only: it is
not exposed to the actor and has no execution authority outside the training
label path.

The first local v6 rollout exposed an important frame-of-reference error rather
than a bad offline fit. Error fell to 0.1040 m at step 30, then diverged as the
moving pelvis repeatedly changed the polynomial input: 7.27% of feature samples
and 9.12% of normalized posture actions were clamped, final mean error reached
0.2229 m, and final p90 reached 0.6880 m. The static map had accidentally become
a second feedback controller coupled to base sway.

V7 evaluates the posture map exactly once from each episode's initial typed
wrist target and stores the resulting per-environment joint posture. The cached
posture chooses the kinematic branch; adaptive DLS alone responds to subsequent
base motion and actuator lag. Reset clears the cache and all associated clamp
diagnostics. This preserves target conditioning without feeding dynamic pelvis
error back through a model fitted only on the nominal stance.

The cached v7 run removed feature extrapolation entirely but did not remove the
failure: minimum mean error was 0.1003 m, while the full-gain DLS term still
drove outliers away after step 30 and final p90 reached 0.7200 m. V8 therefore
uses the fitted posture itself as the bounded servo target. The posture term has
unit gain before the existing 0.20 rad correction, 0.16 rad lead and 0.03 rad
slew limits; task-space feedback gain is explicitly zero. Jacobian/DLS work is
temporarily retained only as a diagnostic probe, so this rollout isolates the
validated feed-forward map from the controller that has repeatedly selected the
wrong dynamic branch.

V8 then proved that the kinematic posture could not be tracked by the source
locomotion plant: success remained zero, minimum mean error regressed to
0.1551 m, both shoulder-roll lead limits were active on every sample, and joint
effort reached its hardware limit. V9 kept the lower body and waist exactly
teacher-aligned but applied the Harness production arm/wrist stiffness to the
fourteen residual-owned joints. This was directionally useful: minimum mean
error improved to 0.1201 m and upper-body peak velocity fell from 3.18 to
2.40 rad/s while post-settle support margin remained positive at 0.0432 m.
It was not sufficient: success remained zero, both shoulder-roll lead limits
remained fully occupied, and final mean error rebounded to 0.2055 m. Colab,
DAgger, and PPO therefore remain gated.

V10 retains the v9 ownership-partitioned plant and restores only 0.1 of the DLS
task-space correction, versus the unstable 1.0 used by v7. The posture map still
selects the kinematic branch; low-gain measured-state feedback is responsible
only for plant lag and base motion. Evaluation also reports per-joint actuator
force utilization and measured target-tracking error so a torque-limited pose
cannot be misdiagnosed as another IK failure.

The v10 screen rejected that hypothesis. Minimum mean error changed only from
0.1201 to 0.1191 m, while final p90 worsened from 0.4381 to 0.6456 m. Per-joint
diagnostics instead showed 0.10--0.12 rad shoulder-roll RMS tracking error at
only 53% of the hardware effort limit. This is consistent with gravity-loaded
PD steady-state error, not torque saturation: unit posture gain commands the
geometric pose itself and supplies no target offset to carry the arm's static
load.

V11 removes DLS feedback again and raises only the measured-state posture servo
gain to 2.5. The resulting joint command may lead the geometric posture enough
to cancel steady load, but remains inside the existing 0.20 rad correction,
0.16 rad command-lead, soft-limit, Harness-authority, and Unitree effort bounds.
This tests gravity-bias rejection without adding an actuator, feed-forward torque
channel, or action dimension.

V11 also failed that test. Minimum mean error improved by only 0.0031 m over v9,
both shoulder-roll tracking errors remained unchanged, final p90 grew to
0.7132 m, and foot slip nearly doubled. V12 returns to v9's unit posture gain and
zero DLS feedback, then adds no behavior change: it records generalized
constraint force and bias force per joint alongside actuator effort. Those
signals distinguish a collision or joint-limit reaction from gravity/Coriolis
load before the teacher architecture is changed again.

V12 localized the blocker to the constraint solver. Left/right shoulder-roll
constraint reactions peaked at 19.88/16.19 Nm while their bias loads were only
1.75/1.92 Nm and actuator use remained 54%/53% of the hardware limit. V13 keeps
the v12 behavior unchanged and adds diagnostic-only contact sensors for every
upper-body link against any collider, the torso, the rod, and the pickup stand.
These sensors are absent from actor and critic observations; they exist only to
name the collision source that invalidated the unconstrained offline IK map.

The v13 sensors identified the missing physical constraint. The old
collision-free-in-name-only IK route drove `left_shoulder_yaw_link` and
`right_shoulder_yaw_link` through the torso, with peak forces near 232 N and
179 N and rollout contact occupancy near 54% and 42%. Wrist and finger links
could also strike the rod before the authorized contact phase. The earlier
postures were mathematically reachable but not physically admissible.

## Collision-aware reach routing and accepted v15 teacher

The correction is an Agent Harness authority decision as well as a kinematic
one. While waist authority remains locked, a low-level reach executor may not
select a cross-torso route merely because unconstrained IK reports it as short.
The typed command now assigns the laterally nearest hand: objects on the left
use the left arm, objects on the right use the right arm, and exact centerline
samples are balanced. The shoulder-ray pre-grasp retains its longitudinal and
vertical offset while enforcing same-side lateral clearance.

`training/fit_workyard_reach_posture.py` makes the posture prior reproducible.
It fits bounded per-arm quadratic maps from 5-by-5 MuJoCo samples and validates
them independently on 17-by-17 grids. Its objective penalizes torso,
contralateral-arm, rod, and stand collisions. The resulting frozen map is still
teacher supervision only: it is absent from actor observations and has neither
runtime execution authority nor a path to hands, waist, or locomotion joints.

V14 used 0.08 m lateral clearance with the collision-aware map and the v9
ownership-partitioned plant. It was the first dynamic teacher to cross the
distance threshold: a local 16-by-200 rollout reached 13/16 (81.25%) success,
0.05771 m minimum mean wrist error, zero falls, zero feature/action clamps,
0.04579 m post-settle support margin, and 0.07069 m/s peak foot slip. The three
failures were no longer torso penetrations; wrist or hand geometry touched the
rod early. V14 therefore remained below the mandatory 85% teacher gate.

A 0.12 m candidate put every dense-grid wrist error below 0.06 m but reduced
the conservative 5 mm collision-clearance pass rate to about 84.08%, so it was
rejected before dynamic rollout. V15 selects 0.10 m instead. Its independent
collision-aware validation passes 513/578 points (88.754%) at 5 mm clearance,
while all 578 points remain within 0.06 m wrist error. Mean, conservative p90,
and maximum errors are 0.008503 m, 0.018600 m, and 0.034634 m. The remaining
clearance failures have zero penetration and only miss the conservative 5 mm
margin.

The fixed-seed local RTX 3050 Ti v15 screen then reached 16/16 success with
0.05771 m minimum mean error, zero falls, zero non-foot termination, and zero
action or feasible-posture clamps. Post-settle support margin was 0.06543 m and
peak foot slip was 0.01240 m/s. Torso contacts fell to isolated samples: left
and right shoulder-yaw contact rates were 0.0625% and 0.09375%, rather than the
long-lived v13 collisions.

The formal Colab Pro L4 teacher gate repeated the result across 128 environments
and 200 control steps. It reached 128/128 success, split 69/69 left and 59/59
right, with 0.05655 m minimum mean error, 0.05996 m maximum final error, zero
falls, zero non-foot termination, zero action clipping, and zero posture feature
or normalized-action clamps. Post-settle support margin remained 0.06443 m and
peak foot slip 0.01903 m/s. The analytic-teacher preflight requirement is now
satisfied; DAgger may begin. This does not authorize hand contact/grasp or waist
checkpoint expansion. Those remain gated on the trained policy's separate
500-episode held-out reach and dynamic-CoM acceptance.

The first v15 learner run used 2,048 parallel environments, 1,000 online
DAgger control steps, and 2,048,000 finite teacher labels. Supervised loss fell
from 0.10836 to 0.000527, final bounded-policy action standard deviation was
0.1373, and teacher label coverage was 100%. On the independent 500-by-600
held-out rollout, the DAgger checkpoint reached 500/500 wrist successes with
0.05665 m minimum mean error, zero falls, zero action clipping, 0.06416 m
minimum post-settle support margin, and 0.03600 m/s peak foot slip. This is an
autonomous actor checkpoint: the analytic posture map is not an observation or
a runtime execution path.

The same run deliberately evaluated a retained-PPO candidate after 100 updates
and 6,553,600 environment steps. Although every stored rollout state had a
finite teacher label and final mean action standard deviation fell to 0.1027,
cumulative actor parameter drift reached 7.64%. The PPO checkpoint collapsed to
0/500 success, raised minimum mean wrist error by 0.03169 m, reduced support
margin by 0.01682 m, and raised peak foot slip by 0.01751 m/s. The no-regression
retention gate rejected it and selected the byte-identical DAgger checkpoint.
This is the intended hybrid outcome: imitation owns the solved free-space reach
primitive; PPO remains optional and must earn replacement authority rather than
being forced into every layer.

That run also exposed an acceptance bookkeeping defect. Checkpoint selection
copied the DAgger file after rejecting PPO, but the top-level evaluation still
referenced the in-memory PPO policy. Selection now returns the evaluation that
matches the selected checkpoint. Phase-one acceptance checks that safe
selection: a retained PPO may pass, or a rejected PPO must have an explicit
rollback to DAgger. An unselected PPO candidate no longer blocks a DAgger reach
checkpoint that independently passes all 500 held-out reach and dynamic-CoM
thresholds. PPO retention status remains separately reported and never grants
authority by itself.

The corrected qualification run repeated the full 2,048-environment schedule
and independently confirmed the safe-selection path. The selected and DAgger
checkpoint SHA-256 values are identical. Its 500 held-out episodes split
237/237 left and 263/263 right successes, with 0.05669 m minimum mean wrist
error, 0.05998 m maximum final error, zero falls, zero action clipping,
0.06404 m minimum post-settle support margin, 0.04998 m maximum capture-point
norm, 0.00102 m maximum foot displacement, and 0.02859 m/s peak foot slip.
The report marks the selected-checkpoint safety gate and phase-one acceptance
as passed, authorizes hand checkpoint expansion, and keeps waist expansion
locked. The Colab session was terminated after the artifacts and report were
downloaded.
