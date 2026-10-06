<script setup>
import { ref, onMounted } from "vue";
import { getResults } from "../api.js";

const rows = ref([]);
const loading = ref(true);

const SCENARIO_ORDER = { lt: 0, eq: 1, gt: 2 };

onMounted(async () => {
  const all = await getResults("add_states");
  rows.value = all
    .filter((r) => r.algo === "ppo_mask")
    .sort((a, b) => (SCENARIO_ORDER[a.scenario] ?? 9) - (SCENARIO_ORDER[b.scenario] ?? 9));
  loading.value = false;
});

function fmtPct(x) {
  return x === null || x === undefined ? "—" : (x * 100).toFixed(1) + "%";
}
function fmt4(x) {
  return x === null || x === undefined ? "—" : Number(x).toFixed(4);
}
function fmtGap(ilp, rl) {
  if (ilp === null || ilp === undefined || rl === null || rl === undefined) return "—";
  return (ilp - rl).toFixed(4);
}
</script>

<template>

  <article class="paper">
    <header class="paper-header">
      <div class="paper-kicker">Work log · add_states branch</div>
      <h1 class="paper-title">Reinforcement Learning for Service-to-ECU Placement: AR-Oriented Reward Design and the Capability Limit</h1>
      <div class="paper-meta">August 2026 · MaskablePPO (P4) · lt / eq / gt scenarios</div>
    </header>

    <section class="paper-abstract">
      <h2 class="paper-abstract-label">Abstract</h2>
      <p>
        This document records a series of diagnoses and experiments on the service-to-ECU placement problem
        (formulated as an ILP; RL uses action masking to guarantee zero violations), aimed at two goals:
        <strong>low success_rate</strong> and <strong>bringing AR (resource utilization) close to the ILP optimum</strong>.
        There are three main findings:
        (1) the effect of BC pre-training splits by how scarce resources are — harmful in the scarce scenario (lt), helpful in the abundant ones (eq/gt);
        (2) eq/gt had never fed AR into the training objective; porting the AR-proportional reward cut the AR gap from
        0.03&nbsp;~&nbsp;0.17 to 0.004&nbsp;~&nbsp;0.05, the largest single change found so far;
        (3) for lt, where success_rate is stuck at 80%&nbsp;~&nbsp;90%, five consecutive independent reward-engineering
        attempts (potential-based shaping, weight rebalancing, entropy annealing, graded failure penalty, feasibility shaping) all had zero effect,
        indicating that this is the real capability limit of the current "online, single-step, non-backtracking" decision structure under pure reward learning,
        not an untuned reward.
      </p>
    </section>

    <nav class="paper-toc">
      <strong>Contents</strong>
      <ol>
        <li><a href="#sec-bg">Background</a></li>
        <li><a href="#sec-method">Method</a></li>
        <li><a href="#sec-exp1">Experiment 1: scenario-dependent effect of BC pre-training</a></li>
        <li><a href="#sec-exp2">Experiment 2: five rounds of reward-engineering ablation</a></li>
        <li><a href="#sec-results">Final results</a></li>
        <li><a href="#sec-discussion">Discussion: capability limit vs. Safe RL</a></li>
        <li><a href="#sec-conclusion">Conclusion and outlook</a></li>
      </ol>
    </nav>

    <section id="sec-bg">
      <h2>1. Background</h2>
      <p>
        The task is to assign <em>M</em> services to <em>N</em> ECUs. Each ECU has a capacity constraint, and services have
        conflict-set constraints (services in the same conflict set cannot share an ECU). The three scenarios correspond to three supply–demand relations:
        <strong>lt</strong> (N&lt;M, scarce, fewer ECUs than services), <strong>eq</strong> (N=M, balanced),
        <strong>gt</strong> (N&gt;M, abundant). The ILP (Dinkelbach parametric subproblem) gives the global optimum,
        used as the upper bound RL tries to approach. The core metric AR (Average Resource Utilization) is the sum of
        "demand / capacity" over the used ECUs divided by the number of used ECUs.
      </p>
      <p>
        On the RL side, MaskablePPO (denoted P4) uses action masking to make the capacity and conflict constraints
        <strong>hard constraints</strong> — illegal actions are removed from the action space, so the violation rate is always 0 in training and evaluation.
        This makes "can all M services be placed legally" (success_rate) and "how high is the utilization afterwards" (AR)
        two independently measurable goals.
      </p>
    </section>

    <section id="sec-method">
      <h2>2. Method</h2>
      <h3>2.1 Evolution of the reward design</h3>
      <p>
        The reward went through three stages: <strong>v1.1.0</strong> purely terminal sparse reward (success <code>+M</code> /
        violation <code>-M</code>, no distinction by AR) → <strong>v2.2.0</strong> AR becomes the training objective
        (success <code>M·AR</code> / violation <code>-M</code>, first on lt) →
        <strong>v2.4.0/v2.5.0</strong> (the main subject here) equal weight for AR and violations (<code>M·(2·AR-1)</code>) +
        failures graded by <code>valid_placed/M</code>, replacing the earlier 0/1-loss design where
        "every failure is the same <code>-M</code>".
      </p>
      <h3>2.2 State design: bottleneck_risk</h3>
      <p>
        The observation gains one aggregate feature, <code>bottleneck_risk</code> — the mean of
        <code>1/(valid_ecu_count+1)</code> over the remaining services, a continuous "approaching a dead end" signal, so the policy
        no longer has to infer the risk implicitly from per-service raw counts of legal ECUs.
      </p>
    </section>

    <section id="sec-exp1">
      <h2>3. Experiment 1: scenario-dependent effect of BC pre-training</h2>
      <p>
        An isolated experiment (same 5M steps, only BC switched) shows: on <strong>lt</strong>, removing BC raises
        success_rate from 47.5% to 70.0% (+22.5pp, AR almost unchanged), the largest single negative factor found so far;
        but on <strong>eq/gt</strong> (success_rate already capped at 100%), removing BC widens the
        AR gap from about 0.03 / 0.14 to 0.076 / 0.173 — BC is actually a net positive when resources are abundant.
      </p>
      <table class="paper-table">
        <thead><tr><th>Scenario</th><th>Setting</th><th>success_rate</th><th>AR gap</th></tr></thead>
        <tbody>
          <tr><td rowspan="2">lt</td><td>with BC (old baseline)</td><td>47.5%</td><td>0.089</td></tr>
          <tr><td>without BC</td><td>70.0%</td><td>0.087</td></tr>
          <tr><td rowspan="2">eq</td><td>with BC</td><td>100%</td><td>≈0.03</td></tr>
          <tr><td>without BC</td><td>100%</td><td>0.076</td></tr>
          <tr><td rowspan="2">gt</td><td>with BC</td><td>100%</td><td>≈0.14</td></tr>
          <tr><td>without BC</td><td>100%</td><td>0.173</td></tr>
        </tbody>
      </table>
      <p class="paper-note">Conclusion: the BC policy splits by resource scarcity — disabled on lt, kept on eq/gt (v2.3.0).</p>
    </section>

    <section id="sec-exp2">
      <h2>4. Experiment 2: five rounds of reward-engineering ablation (lt)</h2>
      <p>
        To address lt's success_rate / AR being stuck in a band, five independent, theoretically motivated
        reward changes were tried in a row, each validated with 8&nbsp;~&nbsp;10 random seeds (not a single run).
      </p>
      <table class="paper-table">
        <thead><tr><th>#</th><th>Change</th><th>success_rate</th><th>AR mean</th><th>Verdict</th></tr></thead>
        <tbody>
          <tr><td>0</td><td>Baseline: M·AR reward, no entropy annealing</td><td>86.50% ± 5.68</td><td>0.6207 ± 0.0074</td><td>—</td></tr>
          <tr><td>1</td><td>bottleneck_risk potential shaping (discrete → continuous, several β)</td><td>85~87%</td><td>0.6207~0.6208</td><td>no effect</td></tr>
          <tr><td>2</td><td>reward stretched to M·(2·AR-1) + entropy annealing 0.02→0.002</td><td>87.19% ± 6.19</td><td>0.6276 ± 0.0100</td><td>no effect</td></tr>
          <tr><td>3</td><td>failure penalty graded by valid_placed/M</td><td>85.62% ± 5.13</td><td>0.6210 ± 0.0132</td><td>no effect</td></tr>
          <tr><td>4</td><td>shaping potential replaced by an FFD greedy feasibility simulation</td><td>85.50% ± 4.53</td><td>0.6208 ± 0.0134</td><td>no effect</td></tr>
        </tbody>
      </table>
      <p class="paper-note">
        Five design ideas with very different signal strengths all converge to the same band (success_rate 80~90%,
        AR 0.60~0.65); the differences are within seed noise (about 5~6pp).
      </p>
    </section>

    <section id="sec-results">
      <h2>5. Final results</h2>
      <p>
        With the branch in its current state (all changes recorded here merged), one standard training run per scenario
        (lt 5,000,000 steps, eq/gt 2,000,000 steps each, no BC). The table below is read live from the latest run in
        <code>results/add_states/&lt;scenario&gt;/ppo_mask/</code>.
      </p>
      <div v-if="loading" class="empty">Loading…</div>
      <table v-else class="paper-table">
        <thead>
          <tr><th>Scenario</th><th>N</th><th>M</th><th>ILP AR</th><th>RL AR</th><th>AR gap</th><th>success_rate</th><th>Training steps</th></tr>
        </thead>
        <tbody>
          <tr v-for="r in rows" :key="r.scenario">
            <td>{{ r.scenario }}</td>
            <td>{{ r.N }}</td>
            <td>{{ r.M }}</td>
            <td>{{ fmt4(r.ilp_ar) }}</td>
            <td>{{ fmt4(r.test_ar_mean) }}</td>
            <td>{{ fmtGap(r.ilp_ar, r.test_ar_mean) }}</td>
            <td>{{ fmtPct(r.test_success_rate) }}</td>
            <td>{{ r.train_steps?.toLocaleString() ?? "—" }}</td>
          </tr>
          <tr v-if="!rows.length"><td colspan="8" class="empty">No data yet — training may still be running; refresh this page when it finishes.</td></tr>
        </tbody>
      </table>
    </section>

    <section id="sec-discussion">
      <h2>6. Discussion: capability limit vs. Safe RL</h2>
      <p>
        The capacity/conflict constraints can be made hard (masking) because they are <strong>step-wise, locally verifiable</strong>
        properties; "can all services be placed" (success) is a <strong>global property of the whole trajectory</strong>
        and cannot be guaranteed by a one-step check. In principle masking could be extended to a "look-ahead feasibility check" (verifying at every step
        that the remaining subproblem is still solvable), making success a hard constraint too, but then
        <strong>success_rate would no longer measure "what RL learned" but "how strong the solver embedded in the environment is"</strong>
        — the same issue as the previously rejected "ILP fallback" scheme, just hidden somewhere else.
      </p>
      <p>
        Using solver signals for <em>reward shaping</em> (rather than masking) is consistent with "RL learns autonomously" — the solver
        only gives a more precise teaching signal during training; the policy still has to learn how to respond, and deployment does not depend on the solver.
        Round 4 of Experiment 2 (FFD feasibility shaping) was exactly such an experiment, and it still had zero effect,
        further supporting the conclusion that "this is the capability limit of the decision structure itself".
      </p>
    </section>

    <section id="sec-conclusion">
      <h2>7. Conclusion and outlook</h2>
      <p>
        Within the framework of "RL learns entirely through the reward", lt's success_rate of 80%&nbsp;~&nbsp;90% and
        AR of 0.60&nbsp;~&nbsp;0.65 are the real capability limit of the current online, single-step, non-backtracking decision structure, not
        an untuned reward — a conclusion based on five independent multi-seed experiments rather than single runs.
        eq/gt, after adding the AR reward, have already cut the AR gap to 0.004&nbsp;~&nbsp;0.05,
        very close to the ILP optimum.
      </p>
      <p>
        Directions worth exploring next: GRPO-style group-relative advantage training (multiple samples per scenario, compared within the group rather than averaged across scenarios),
        curriculum learning (oversampling scenarios with high demand/capacity ratio), and look-ahead/search during training
        (AlphaZero-like: search results only generate training targets and are not needed at deployment).
      </p>
    </section>

    <footer class="paper-footer">
      Full version history in <a href="#/versions">Version History</a>
      (<a href="#/versions/v2.3.0">v2.3.0</a>,
      <a href="#/versions/v2.4.0">v2.4.0</a>,
      <a href="#/versions/v2.5.0">v2.5.0</a>).
    </footer>
  </article>
</template>
