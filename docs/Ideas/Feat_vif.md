# Value Identity Function (VIF) – Technical Design (Capstone-Friendly)

This section specifies the mathematical and system design for Twinkl’s core evaluative engine: the **Value Identity Function (VIF)**. VIF estimates how well a user’s recent and projected behaviour aligns with their long-term values and identity, and underpins the system’s “timely critique” and “tension surfacing” capabilities.

This version presents a **tiered design** suitable for an academic proof-of-concept (POC), with multiple implementation options (from simple to advanced). The capstone team can choose which tier to implement while keeping a coherent long-term architecture.

---

## 1. Objectives and Design Principles

### 1.1 Objectives

The VIF is designed to:

* Estimate **per-value-dimension alignment** for a user at a given point in time.
* Detect **emerging misalignment trajectories** early (before long-term regret).
* Support **explainable feedback**: not just “good/bad,” but “which parts of your life are drifting and why.”
* Operate safely on **real user data**, where no ground-truth reward labels exist.

### 1.2 Key Design Principles

1. **Reward modeling, not oracle reward**
   The system does not assume access to ground-truth alignment labels. Instead, it uses an explicit **Reward Model (RM)** to infer alignment scores from text and user profiles.

2. **Vector-valued evaluation**
   Alignment is evaluated in **multiple value dimensions** (e.g. Health, Relationships, Growth, Contribution). The value function remains **vector-valued** to preserve tensions and trade-offs, and is only aggregated when needed.

3. **Uncertainty-aware critique**
   The system estimates **epistemic uncertainty** in its predictions and only issues strong critiques when it is both:

   * Confident in its judgment, and
   * Detecting a significant negative pattern.

4. **Trajectory-aware, not purely Markovian**
   The VIF explicitly incorporates **recent temporal history** (sliding windows and simple statistics) instead of assuming that a single entry and static profile fully determine future trajectories.

5. **Separation of concerns: Critic vs Coach**
   The VIF (critic) focuses on **temporal, numeric evaluation** using sequential history. A separate **Coach / Explanation layer** uses retrieval-augmented generation (RAG) to surface thematic evidence and narratives *after* the critic has produced its scores.

---

## 2. Value Dimensions and User Profile

### 2.1 Value Dimensions

We define a set of (K) value dimensions:

* Example: Health, Relationships, Growth, Contribution, Autonomy, etc.
* Each value dimension has:

  * A **definition** in natural language.
  * Positive and negative **examples**.
  * A **rating rubric** (e.g. from “strongly misaligned” to “strongly aligned”).

These dimensions form an ontology for both the reward model and the critic.

### 2.2 User Value Profile

Each user (u) has a **value profile**:

* A vector of value weights:

  * (w_{u,t} \in \mathbb{R}^K), with (w_{u,t} \ge 0) and (\sum_k w_{u,t,k} = 1).
  * For the POC, (w_{u,t}) may be treated as **piecewise constant** over time (updated infrequently), but conceptually it can evolve slowly as the user refines their priorities.
* Additional profile information:

  * Narrative descriptions of what each value means to them.
  * Known constraints and long-term goals.

This profile is used both in the **Reward Model prompts** and in aggregating vector-valued outputs for summaries.

---

## 3. Inputs and State Representation

### 3.1 Raw Inputs

For each user (u) at time step (t), we assume:

* Text journal or transcript: (T_{u,t}).
* Optional audio: (A_{u,t}) (voice recordings, prosody).
* Optional physiological signals: (H_{u,t}) (e.g. heart rate, HRV).
* Time features:

  * (\Delta t_{u,t} = t_{u,t} - t_{u,t-1}): time since previous entry.
* User profile:

  * (z_u): embedding or feature vector representing the user’s values and identity.

### 3.2 Embedding Functions

We define embedding functions:

* Text encoder (e.g. SBERT or similar):
  (\phi_{\text{text}}(T_{u,t}) \in \mathbb{R}^{d_e})
* Optional audio/prosodic encoder:
  (\phi_{\text{audio}}(A_{u,t}) \in \mathbb{R}^{d_p})
* Optional physiological encoder (e.g. via time-series feature extraction):
  (\phi_{\text{physio}}(H_{u,t}) \in \mathbb{R}^{d_h})
* User profile embedding:
  (z_u \in \mathbb{R}^{d_z})

### 3.3 State Vector – Sliding Window Design

#### 3.3.1 Motivation: Beyond the Markov Assumption

A design that uses only the current entry (T_{u,t}) and static profile (z_u) implicitly assumes a **Markov state**: all relevant information for future alignment is contained in the present features. In real life, this is false. A single entry of “I’m exhausted” may be noise; three consecutive entries with the same theme form a **trajectory**.

To capture trajectories without complex recurrent architectures in the critic, we adopt a **sliding window** of recent entries.

#### 3.3.2 Sliding Window State Definition (Core POC Option)

For a window size (N) (e.g. (N = 3) for the POC):

[
s_{u,t} = \text{Concat}\Big[
\phi_{\text{text}}(T_{u,t}),
\phi_{\text{text}}(T_{u,t-1}),\ \dots,
\phi_{\text{text}}(T_{u,t-N+1}),
\Delta t_{u,t},\dots,\Delta t_{u,t-N+2},
\text{history stats},
z_u
\Big]
]

Where **history stats** may include simple per-dimension statistics derived from past Reward Model outputs (for a chosen lookback horizon):

* Exponential moving averages (EMA) of per-dimension alignment.
* Rolling standard deviations.
* Counts of entries over recent time windows (e.g. last 7 days, last 30 days).

For early time steps, where fewer than (N) entries exist, we can:

* Pad missing embeddings with zeros, or
* Use special “no-history” tokens/flags.

#### 3.3.3 Minimal MVP State (Simplified Option)

For the simplest POC, the state can initially be:

[
s_{u,t} = \text{Concat}\big[\phi_{\text{text}}(T_{u,t}),\ z_u\big]
]

with the sliding window and history stats added in a later iteration.

#### 3.3.4 Multimodal Extension (Future Option)

Audio and physiological channels can be appended once available:

[
s_{u,t} = \text{Concat}\Big[\text{(text window)},\ \phi_{\text{audio}}(A_{u,t}),\ \phi_{\text{physio}}(H_{u,t}),\ z_u,\ \text{history stats}\Big]
]

---

## 4. Reward Modeling (LLM-as-Judge)

### 4.1 Problem: No Oracle Reward in Real Data

For real users, we do not observe:

* True per-dimension alignment (a_{u,t}^{(j)}) (e.g. “Health = −0.8”).
* True cumulative returns.

We only observe **text and signals**. To bridge this, we explicitly introduce a **Reward Model**.

### 4.2 LLM-as-Judge

We use a large language model (LLM) as a **judge**.

For each time step ((u,t)), the LLM is given:

* The journal text (T_{u,t}).
* The user’s profile (z_u) and current value weights (w_{u,t}).
* The definitions and rubrics for the (K) value dimensions.

The LLM outputs an alignment score per value dimension:

[
\hat{a}*{u,t}^{(j)} = \text{LLM}*{\text{Judge}}\big(T_{u,t}, z_u, w_{u,t}, \text{dimension } j\big)
\quad \text{for } j = 1, \dots, K
]

These scores are:

* Typically on a Likert scale (e.g. −3 to +3), normalized to ([-1,1]).
* Accompanied by textual rationales.
* Optionally accompanied by a **self-reported confidence** score from the LLM (e.g. 1–5), which can be used to weight training samples.

We denote the vector of per-dimension alignment scores as:

[
\hat{\vec{a}}*{u,t} = (\hat{a}*{u,t}^{(1)}, \dots, \hat{a}_{u,t}^{(K)})
]

To reduce normative bias, prompts are explicitly framed to judge **alignment relative to the user’s stated values and trade-offs**, not any external cultural or moral standard.

### 4.3 Label Quality and Synthetic Data

In early stages, we may use **synthetic personas and trajectories** where ground-truth per-dimension alignment is known, and the LLM-as-Judge is used for comparison and distillation. Later, for real user data, LLM labels form a noisy but useful signal.

To characterise label noise and calibration:

* On a subset of entries, run the judge multiple times or with slight prompt variations to estimate label variance.
* Optionally, incorporate this variance or self-reported confidence into the critic’s loss as weights.

---

## 5. Vector-Valued Value Function – Multiple Target Options

### 5.1 Motivation

Scalarizing rewards too early (e.g. (r = w^\top a)) hides important conflicts:

* A week with (+10) Career and (−10) Relationships is not “neutral” in human terms.
* Twinkl’s purpose is to **surface such tensions**, not to flatten them away.

Therefore, we keep the value function **vector-valued**.

We define several **target options** for the VIF, from simplest to most advanced. The capstone team can choose one for implementation.

### 5.2 Option A: Immediate Alignment (Simplest)

The VIF directly predicts the Reward Model’s immediate alignment scores:

[
\vec{V}*\theta(s*{u,t}) \approx \hat{\vec{a}}_{u,t}
]

This is a **multi-output regression** problem, where the critic learns to emulate the judge, potentially with:

* Lower latency.
* Better generalisation from richer state features (e.g. using history window and profile).

Longer-term trends are then derived via **deterministic smoothing** (e.g. EMA over time), not via discounted returns.

### 5.3 Option B: Short-Horizon Forecast (Intermediate)

The VIF predicts short-horizon average alignment, e.g. over the next (H) days or entries (e.g. (H = 7) days):

[
G_{u,t}^{(j, H)} = \frac{1}{H'} \sum_{k=0}^{H'-1} \hat{a}_{u,t+k}^{(j)}
]

where (H') is the number of future entries within the horizon. The critic is trained to predict:

[
\vec{V}*\theta(s*{u,t}) \approx \vec{G}*{u,t}^{(H)} = (G*{u,t}^{(1,H)}, \dots, G_{u,t}^{(K,H)})
]

This gives genuinely **prospective** signals but limits the horizon to a more realistic, data-supported window.

### 5.4 Option C: Discounted Returns (Advanced)

For each user (u), time (t), and value dimension (j):

1. The Reward Model provides immediate alignment: (\hat{a}_{u,t}^{(j)} \in [-1,1]).
2. We define a **discounted cumulative alignment** (return):

[
G_{u,t}^{(j)} = \sum_{k=0}^{T_u - t} \gamma^{g(\Delta t_{u,t+k})} \hat{a}_{u,t+k}^{(j)}
]

where:

* (\gamma \in (0,1]) is the base discount factor.
* (g(\Delta t)) is a function that maps irregular time gaps to effective discount exponents (e.g. (g(\Delta t) = \Delta t / \tau) for some time constant (\tau)).
* (T_u) is the final time step for user (u).

The critic approximates:

[
\vec{V}*\theta(s*{u,t}) \approx \mathbb{E}[\vec{G}*{u,t} \mid s*{u,t}]
]

This option is closest to an RL-style value function but also the most demanding.

### 5.5 Scalar Aggregation (for Summaries Only)

When needed (e.g. for an overall weekly alignment score), we can aggregate:

[
V^{\text{scalar}}*{u,t} = w*{u,t}^\top \vec{V}*\theta(s*{u,t})
]

This scalar is used only for **summary views**; critiques and explanations are primarily based on the vector (\vec{V}_\theta).

---

## 6. Training Objective and Strategy

### 6.1 Training Type

For the capstone POC, training the critic is:

* **Supervised learning**, not reinforcement learning.
* The model learns a mapping:

  * From state (s_{u,t})
  * To a chosen target vector (depending on Option A/B/C): (\hat{\vec{a}}*{u,t}), (\vec{G}*{u,t}^{(H)}), or (\vec{G}_{u,t}).

We use:

* **Pretrained models** (e.g. SBERT) for text embeddings.
* A **from-scratch multi-output regressor** for the critic itself.

### 6.2 Loss Function

For a generic target vector (\vec{Y}_{u,t}) (immediate, short-horizon, or discounted), we define the loss over all users and time steps:

[
\mathcal{L}(\theta) = \frac{1}{|\mathcal{D}|} \sum_{(u,t) \in \mathcal{D}} \sum_{j=1}^K w_{u,t}^{\text{label}} \Big( V_\theta^{(j)}(s_{u,t}) - Y_{u,t}^{(j)} \Big)^2
]

where:

* (\mathcal{D}) is the dataset of all (state, target) pairs.
* (w_{u,t}^{\text{label}}) is an optional **label-weighting term** (e.g. based on judge confidence or label variance estimates).

In the simplest version (Option A), (\vec{Y}*{u,t} = \hat{\vec{a}}*{u,t}) and (w_{u,t}^{\text{label}} = 1).

### 6.3 Model Architecture

A minimal architecture:

* Input: state vector (s_{u,t}).
* Hidden layers: 2–3 dense layers (MLP) with non-linearities (e.g. ReLU or GELU), with dropout for regularisation and uncertainty estimation.
* Output layer: linear layer with (K) outputs (one per value dimension).

The text encoder (e.g. SBERT) is:

* Either frozen in the first iteration (for simplicity and data efficiency).
* Or lightly fine-tuned jointly with the critic if data allows.

### 6.4 Training Pipeline Overview

1. **Ontology creation**: define value dimensions, rubrics, and examples.
2. **Reward modeling (offline)**:

   * Run LLM-as-Judge on existing (synthetic and/or real) entries to obtain (\hat{\vec{a}}_{u,t}) and optional confidence scores.
3. **Target construction**:

   * Depending on chosen option:

     * Option A: set (\vec{Y}*{u,t} = \hat{\vec{a}}*{u,t}).
     * Option B: compute short-horizon averages (\vec{G}_{u,t}^{(H)}).
     * Option C: compute discounted returns (\vec{G}_{u,t}) with time-aware discounting.
4. **State construction**:

   * Encode text (and other signals if available) into windowed state vectors (s_{u,t}).
5. **Critic training**:

   * Train (\vec{V}_\theta) via supervised regression on the chosen targets.
6. **Evaluation**:

   * Use held-out trajectories/personas to evaluate prediction quality (e.g. MSE, correlation) and ranking consistency (e.g. ordering weeks by misalignment).

---

## 7. Uncertainty-Aware Critiques

### 7.1 Problem: Overconfident Critic on OOD Inputs

A critic trained offline (especially on synthetic or limited datasets) can produce **over-confident, incorrect** predictions when faced with out-of-distribution (OOD) inputs (e.g. novel situations or unusual life events).

This is especially problematic in a mental health or values-alignment context.

### 7.2 Monte Carlo Dropout

To estimate **epistemic uncertainty**, we use **Monte Carlo Dropout** at inference time:

1. Keep dropout layers **active** during testing.
2. For a given state (s_{u,t}), run the critic (N) times:

   * (\vec{V}^{(i)} = V_\theta^{(i)}(s_{u,t})), for (i = 1, \dots, N).
3. Compute, for each value dimension (j):

   * Mean (estimate):

[
\mu_{u,t}^{(j)} = \frac{1}{N} \sum_{i=1}^N V^{(i)}_j
]

* Variance (uncertainty):

[
\sigma_{u,t}^{2(j)} = \text{Var}_i\big[V^{(i)}_j\big]
]

The mean (\mu_{u,t}^{(j)}) serves as the value estimate; the variance (\sigma_{u,t}^{2(j)}) is a proxy for epistemic uncertainty.

We can monitor calibration by checking, on held-out data, whether higher (\sigma_{u,t}^{2(j)}) correlates with larger prediction errors.

### 7.3 Dual-Trigger Critique Rule (Crashes and Ruts)

We define two types of problematic patterns for each value dimension (j):

* **Sudden crash:** recent sharp negative change.
* **Chronic rut:** sustained low values over time.

Let:

* (V_{t-1}^{(j)} = \mu_{u,t-1}^{(j)}): previous estimate.
* (V_t^{(j)} = \mu_{u,t}^{(j)}): current estimate.
* (\sigma_{V_t}^{(j)} = \sqrt{\sigma_{u,t}^{2(j)}}): current uncertainty.
* (\delta_j): crash threshold for dimension (j).
* (\tau_{\text{low}}^{(j)}): low-value threshold for dimension (j).
* (C_t^{(j)}): count of **consecutive time windows** (e.g. weeks) where (V_t^{(j)} < \tau_{\text{low}}^{(j)}).

We trigger a **critique** for dimension (j) if **uncertainty is low enough** and either a crash or rut condition is met.

**Condition A – Sudden Crash:**

[
V_{t-1}^{(j)} - V_t^{(j)} > \delta_j
]

**Condition B – Chronic Rut:**

[
V_t^{(j)} < \tau_{\text{low}}^{(j)} \quad \text{and} \quad C_t^{(j)} \ge C_{\text{min}}
]

where (C_{\text{min}}) is a minimum number of consecutive periods (e.g. 3 weeks).

**Uncertainty Constraint:**

[
\sigma_{V_t}^{(j)} < \epsilon_j
]

If the uncertainty is **high** (i.e. (\sigma_{V_t}^{(j)} \ge \epsilon_j)), the system should:

* Avoid issuing a strong critique.
* Prefer asking clarifying questions or deferring judgment.

### 7.4 Time and Aggregation

To implement crash and rut logic robustly:

* VIF outputs can be aggregated to **weekly averages** per dimension.
* Crash/rut detection operates on these weekly aggregates, not raw per-entry values.
* (C_t^{(j)}) is then defined over weeks, making “three weeks in a rut” a natural threshold.

---

## 8. Inference Flow and Memory Architecture

### 8.1 Inference Flow for VIF (Critic)

For a real user session:

1. **Collect input**:

   * User submits a new journal entry (and optional voice/physio signals).
2. **Build sequential state**:

   * Construct sliding window state (s_{u,t}) from the current and (N-1) previous entries, plus profile and history stats.
3. **Reward Model (optional at inference)**:

   * Option A: call LLM-as-Judge online for immediate alignment scores and explanations.
   * Option B: use an offline-trained distilled reward model to approximate judge scores.
4. **Critic evaluation with uncertainty**:

   * Run VIF with MC Dropout to get (\mu_{u,t}^{(j)}) and (\sigma_{u,t}^{(j)}) per dimension.
5. **Aggregate over time**:

   * Compute updated weekly aggregates (e.g. weekly means of (\mu_{u,t}^{(j)})).
6. **Apply dual-trigger critique rule**:

   * If significant weekly drop (crash) and low uncertainty → generate a targeted critique.
   * If sustained low weekly values for (\ge C_{\text{min}}) weeks (rut) and low uncertainty → generate a targeted critique.
   * If high uncertainty → generate a clarifying prompt instead of a critique.

### 8.2 Separation of Concerns: Critic vs Coach

We explicitly separate two memory and reasoning systems:

1. **VIF (Critic)** – Sequential, time-aware evaluator

   * Uses **strict sequential history** (sliding window of recent entries and derived stats).
   * Does **not** use semantic retrieval of arbitrary past entries for its numeric prediction.
   * Focus: estimate current/short-horizon alignment per dimension and uncertainty.

2. **Coach / Explanation Layer** – Thematic, retrieval-augmented guide

   * Activated after the VIF identifies which dimensions are problematic (crash or rut).
   * Uses **retrieval-augmented generation (RAG)** over the user’s longer-term journal history to:

     * Retrieve thematically relevant entries (e.g. similar conflicts, repeated patterns).
     * Provide context-rich explanations and reflective prompts.
   * Retrieval is based on **semantic similarity**, not recency, to support pattern recognition and narrative insight.

This separation ensures that:

* The **numeric value scores** remain grounded in recent temporal dynamics and are not contaminated by arbitrarily distant or thematically similar but contextually irrelevant entries.
* The **explanations and reflections** can draw on the full richness of the user’s history.

---

## 9. Extensions and Future Work

Potential extensions beyond the POC:

* **Distilled Reward Model**:

  * Train a smaller supervised model to mimic the LLM-as-Judge, reducing latency and cost.
* **Policy Learning**:

  * After VIF is stable and safety mechanisms are validated, explore offline RL to learn and evaluate **action suggestions** (micro-experiments, habits). This remains out of scope for the core capstone POC.
* **Richer Uncertainty Modeling**:

  * Incorporate ensembles, density models, or explicit OOD detectors on the text embedding space.
* **More Modalities**:

  * Incorporate prosodic and physiological features robustly, especially for early warning signals of stress or overload.
* **Personalisation Layers**:

  * Explore global VIF plus lightweight per-user adapters for users whose trajectories systematically diverge from the population.

---

## 10. Summary of Implementation Options

To make this design capstone-friendly, we summarise a recommended tiered approach:

* **Tier 1 (Minimum POC)**

  * State: current text embedding + profile (optional small history window).
  * Target: immediate alignment (Option A).
  * Critic: MLP regressor.
  * Uncertainty: MC Dropout.
  * Critique rule: dual-trigger (crash + rut) on simple rolling averages.

* **Tier 2 (Enhanced POC)**

  * State: sliding window over last (N) entries with history stats.
  * Target: short-horizon average (Option B).
  * Critic: MLP with tuned hyperparameters and calibrated uncertainty.
  * Critique rule: weekly crash/rut logic with user-specific thresholds.

* **Tier 3 (Advanced / Future Work)**

  * State: multimodal, sliding-window state with audio/physio.
  * Target: time-aware discounted returns (Option C).
  * Additional: offline RL for suggestion policies, more advanced uncertainty and personalization.

This design keeps the theoretical elegance of a vector-valued, uncertainty-aware value function, while providing pragmatic on-ramps for the capstone team to implement a feasible, defensible POC.
