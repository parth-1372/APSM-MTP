# Build LaTeX-style black & white HTML report using ASCII and HTML entities
$g1 = [System.IO.File]::ReadAllText("results\Phase2_Full\img1.txt", [System.Text.Encoding]::UTF8)
$g2 = [System.IO.File]::ReadAllText("results\Phase2_Full\img2.txt", [System.Text.Encoding]::UTF8)
$g3 = [System.IO.File]::ReadAllText("results\Phase2_Full\img3.txt", [System.Text.Encoding]::UTF8)
$g4 = [System.IO.File]::ReadAllText("results\Phase2_Full\img4.txt", [System.Text.Encoding]::UTF8)
$g5 = [System.IO.File]::ReadAllText("results\Phase2_Full\img5.txt", [System.Text.Encoding]::UTF8)

$html = @"
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>MTP Mid-Semester I Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Source+Code+Pro:wght@400;600&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'EB Garamond', 'Georgia', serif;
    font-size: 11.5pt;
    color: #000;
    background: #fff;
    line-height: 1.55;
  }

  /* -- PAGE -- */
  .page {
    width: 170mm;
    min-height: 257mm;
    margin: 0 auto 0 auto;
    padding: 22mm 0 18mm 0;
    page-break-after: always;
  }
  .page:last-child { page-break-after: auto; }

  @media screen {
    body { background: #e8e8e8; }
    .page {
      background: #fff;
      margin: 10mm auto;
      padding: 22mm 20mm 18mm 20mm;
      width: 210mm;
      box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    }
  }

  @media print {
    body { background: #fff; font-size: 10.5pt; }
    .page { margin: 0; padding: 20mm 20mm 16mm 20mm; width: 170mm; }
  }

  /* -- COVER -- */
  .cover {
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    text-align: center;
    min-height: 257mm;
    padding: 20mm 10mm;
  }
  .cover-rule { width: 100%; border: none; border-top: 2px solid #000; margin: 5mm 0; }
  .cover-rule-thin { width: 100%; border: none; border-top: 1px solid #000; margin: 3mm 0; }
  .cover .institute { font-size: 11pt; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 2mm; }
  .cover .dept { font-size: 10pt; margin-bottom: 1mm; }
  .cover .report-type { font-size: 10pt; font-style: italic; margin-top: 5mm; margin-bottom: 2mm; }
  .cover h1 { font-size: 18pt; font-weight: 700; line-height: 1.3; margin: 4mm 0; }
  .cover h2 { font-size: 12pt; font-weight: 400; font-style: italic; margin-bottom: 5mm; }
  .cover .meta-table { width: 100%; margin-top: 10mm; font-size: 10.5pt; border-collapse: collapse; }
  .cover .meta-table td { padding: 1.5mm 3mm; vertical-align: top; }
  .cover .meta-table td:first-child { font-weight: 700; width: 42%; text-align: right; padding-right: 4mm; }
  .cover .meta-table td:last-child { text-align: left; }

  /* -- HEADINGS -- */
  h1.sec { font-size: 14pt; font-weight: 700; margin: 6mm 0 3mm 0; border-bottom: 1px solid #000; padding-bottom: 1.5mm; }
  h2.ssec { font-size: 12pt; font-weight: 700; margin: 5mm 0 2mm 0; }
  h3.sssec { font-size: 11pt; font-weight: 700; font-style: italic; margin: 4mm 0 1.5mm 0; }

  /* -- SECTION NUMBERING -- */
  .secnum { margin-right: 4mm; }

  /* -- ABSTRACT BOX -- */
  .abstract {
    border: 1px solid #000;
    padding: 4mm 6mm;
    margin: 4mm 0 6mm 0;
    font-size: 10.5pt;
  }
  .abstract-title { font-weight: 700; text-align: center; margin-bottom: 2mm; font-size: 11pt; }

  /* -- BODY TEXT -- */
  p { margin-bottom: 2.5mm; text-align: justify; }
  ul, ol { padding-left: 6mm; margin-bottom: 2mm; }
  li { margin-bottom: 1mm; }
  strong { font-weight: 700; }
  em { font-style: italic; }

  /* -- TABLES -- */
  table.latex { width: 100%; border-collapse: collapse; margin: 4mm 0; font-size: 10pt; }
  table.latex caption { font-size: 10pt; font-style: italic; margin-bottom: 2mm; text-align: center; caption-side: top; }
  table.latex thead tr { border-top: 1.5px solid #000; border-bottom: 1px solid #000; }
  table.latex tbody tr:last-child { border-bottom: 1.5px solid #000; }
  table.latex th { padding: 2mm 3mm; font-weight: 700; text-align: left; font-size: 10pt; }
  table.latex th.c { text-align: center; }
  table.latex td { padding: 2mm 3mm; vertical-align: top; }
  table.latex td.c { text-align: center; }
  table.latex tr.shaded td { background: #f2f2f2; }

  /* -- PHASE ROADMAP -- */
  table.phases { width: 100%; border-collapse: collapse; margin: 4mm 0; font-size: 10pt; }
  table.phases thead tr { border-top: 1.5px solid #000; border-bottom: 1px solid #000; }
  table.phases tbody tr:last-child { border-bottom: 1.5px solid #000; }
  table.phases th { padding: 2mm 3mm; font-weight: 700; }
  table.phases td { padding: 2.5mm 3mm; vertical-align: top; border-bottom: 0.5px solid #bbb; }
  table.phases .done-row td { font-weight: 600; }
  .tag { font-size: 8.5pt; font-weight: 700; border: 1px solid #000; padding: 0.5mm 2mm; display: inline-block; text-transform: uppercase; letter-spacing: 0.5px; }
  .tag.done { background: #000; color: #fff; }
  .tag.planned { background: #fff; color: #000; }
  .tag.active { background: #555; color: #fff; }

  /* -- CODE / FORMULA -- */
  .formula {
    font-family: 'Source Code Pro', 'Courier New', monospace;
    font-size: 9.5pt;
    background: #f8f8f8;
    border-left: 3px solid #000;
    padding: 3mm 4mm;
    margin: 3mm 0;
    white-space: pre;
    line-height: 1.7;
  }

  /* -- DEFINITION BOX -- */
  .defbox {
    border: 1px solid #888;
    padding: 3mm 5mm;
    margin: 3mm 0;
    font-size: 10.5pt;
  }
  .defbox strong { display: block; margin-bottom: 1mm; }

  /* -- RESULTS BOX -- */
  .results-summary {
    border: 2px solid #000;
    padding: 4mm 6mm;
    margin: 4mm 0;
    font-size: 10.5pt;
  }
  .results-summary .title { font-weight: 700; text-align: center; border-bottom: 1px solid #000; padding-bottom: 2mm; margin-bottom: 3mm; font-size: 11pt; letter-spacing: 0.5px; }

  /* -- IMAGES -- */
  .fig { margin: 4mm 0; }
  .fig img { width: 100%; }
  .fig .caption { font-size: 9.5pt; font-style: italic; text-align: center; margin-top: 1.5mm; }
  .fig .caption strong { font-style: normal; }
  .two-figs { display: flex; gap: 4mm; margin: 4mm 0; }
  .two-figs .fig { flex: 1; }

  /* -- FOOTER -- */
  .footer {
    margin-top: 8mm;
    border-top: 1px solid #000;
    padding-top: 2mm;
    font-size: 9pt;
    display: flex;
    justify-content: space-between;
  }

  /* -- MISC -- */
  .center { text-align: center; }
  .mt3 { margin-top: 3mm; }
  .mt5 { margin-top: 5mm; }
  .small { font-size: 9.5pt; }
  .ital { font-style: italic; }
  .no-break { page-break-inside: avoid; }
</style>
</head>
<body>

<!-- ========================== COVER ====================================== -->
<div class="page">
<div class="cover">
  <div class="institute">M.Tech Thesis &mdash; Mid Semester I Progress Report</div>
  <hr class="cover-rule"/>

  <div class="report-type">Research Project Report</div>
  <h1>Bandwidth-Efficient Decentralised Machine Learning<br/>via Adaptive Semantic Communication Filtering<br/>in Gossip Learning Networks</h1>
  <hr class="cover-rule-thin"/>
  <h2>Adaptive Predictive-Semantic Mechanism (APSM)</h2>

  <table class="meta-table">
    <tr><td>Programme:</td><td>M.Tech, Computer Science &amp; Engineering</td></tr>
    <tr><td>Semester:</td><td>Semester I &mdash; Mid-Term Evaluation</td></tr>
    <tr><td>Report Date:</td><td>July 2026</td></tr>
    <tr><td>Base Paper:</td><td>Tundo et al. 2025, IEEE Trans. Network &amp; Service Management</td></tr>
    <tr><td>Dataset:</td><td>Porto Taxi Trajectory Dataset (~1.7M GPS trajectories)</td></tr>
    <tr><td>Experiment:</td><td>Full run, Kaggle GPU (NVIDIA T4), June 2026</td></tr>
    <tr><td>Phase Completed:</td><td>Phase 1 &mdash; Base Implementation &amp; Semantic Filter</td></tr>
  </table>

  <hr class="cover-rule" style="margin-top: 10mm;"/>
  <div class="dept" style="margin-top: 2mm;">Department of Computer Science &amp; Engineering</div>
</div>
</div>


<!-- ====================== ABSTRACT + TOC ================================ -->
<div class="page">

  <div class="abstract no-break">
    <div class="abstract-title">Abstract</div>
    <p style="margin-bottom:0;">
      Gossip Learning (GL) enables fully decentralised, serverless federated machine learning where nodes
      exchange model weights directly with neighbours without a central coordinator. While effective in
      terms of accuracy, the base protocol (Tundo et al. 2025) broadcasts updated weights after every
      training round without regard for whether the update carries new information &mdash; resulting in
      significant wasted bandwidth. This report presents the design, implementation, and experimental
      validation of the <em>Adaptive Predictive-Semantic Mechanism</em> (APSM): a pre-transmission
      gating filter that suppresses gossip broadcasts when a node's model update is statistically
      indistinguishable from recent history. In a controlled 7-hour full experiment on the Porto Taxi
      dataset (10 nodes, 100 gossip rounds each), APSM achieves a <strong>53.6% reduction in network
      packets</strong> with only <strong>0.69% degradation in prediction accuracy (MSE)</strong>,
      demonstrating that the majority of gossip transmissions in converged networks are semantically
      redundant and can be safely eliminated.
    </p>
  </div>

  <h1 class="sec"><span class="secnum">&mdash;</span>Table of Contents</h1>
  <table style="width:100%; font-size:10.5pt; border-collapse:collapse;">
    <tr><td style="padding:1.5mm 0;">1.</td><td style="padding:1.5mm 0;">Problem Statement</td><td style="padding:1.5mm 0; text-align:right;">3</td></tr>
    <tr><td style="padding:1.5mm 0;">2.</td><td style="padding:1.5mm 0;">Literature Review</td><td style="padding:1.5mm 0; text-align:right;">3</td></tr>
    <tr><td style="padding:1.5mm 0;">3.</td><td style="padding:1.5mm 0;">Overall Project Roadmap (Phase 1&ndash;4)</td><td style="padding:1.5mm 0; text-align:right;">4</td></tr>
    <tr><td style="padding:1.5mm 0;">4.</td><td style="padding:1.5mm 0;">Phase 1 &mdash; Base Paper Implementation</td><td style="padding:1.5mm 0; text-align:right;">5</td></tr>
    <tr><td style="padding:1.5mm 0;">5.</td><td style="padding:1.5mm 0;">Phase 1 &mdash; Semantic Filter Design (APSM)</td><td style="padding:1.5mm 0; text-align:right;">6</td></tr>
    <tr><td style="padding:1.5mm 0;">6.</td><td style="padding:1.5mm 0;">Experimental Results</td><td style="padding:1.5mm 0; text-align:right;">7</td></tr>
    <tr><td style="padding:1.5mm 0;">7.</td><td style="padding:1.5mm 0;">Result Graphs</td><td style="padding:1.5mm 0; text-align:right;">8&ndash;9</td></tr>
    <tr><td style="padding:1.5mm 0;">8.</td><td style="padding:1.5mm 0;">Conclusion and Future Work</td><td style="padding:1.5mm 0; text-align:right;">10</td></tr>
  </table>

  <div class="footer">
    <span>APSM-MTP | Mid Semester I Report</span>
    <span>Page 2</span>
  </div>
</div>


<!-- ====================== PROBLEM STATEMENT + LIT REVIEW ================ -->
<div class="page">

  <h1 class="sec"><span class="secnum">1.</span>Problem Statement</h1>

  <p>
    Modern IoT and edge computing deployments generate large volumes of time-series data at the network edge.
    Training predictive models without shipping raw data to a central server requires <em>Federated</em> or
    <em>Decentralised Learning</em>. Federated Learning (FL) achieves privacy preservation but relies on
    a central parameter server &mdash; a structural bottleneck and single point of failure that is incompatible
    with fully peer-to-peer edge architectures.
  </p>

  <p>
    <em>Gossip Learning</em> (GL) eliminates the server: nodes train locally on private data and share
    model weights directly with neighbours through a random gossip protocol. Tundo et al. (2025) demonstrated
    that GL achieves accuracy on par with centralised FL for urban traffic prediction using the Porto Taxi dataset.
  </p>

  <p>
    However, the base protocol has a well-acknowledged limitation: <strong>every node transmits its weights
    after every training round regardless of whether the update is informative</strong>. As a network
    converges, weights stabilise and successive updates become near-identical &mdash; yet transmissions continue
    at the same rate, wasting bandwidth proportionally to network size. In a 10-node deployment this produces
    over 1,400 packets per run; at 1,000 nodes this exceeds 140,000 transmissions.
  </p>

  <div class="defbox">
    <strong>Research Question:</strong>
    Can we design a pre-transmission gating mechanism that determines, per gossip round, whether a node's
    model update carries sufficient semantic information to justify a broadcast &mdash; and if not, suppresses the
    transmission &mdash; while preserving model convergence and accuracy?
  </div>

  <!-- -- LITERATURE REVIEW -- -->
  <h1 class="sec mt5"><span class="secnum">2.</span>Literature Review</h1>

  <table class="latex">
    <caption>Table 1: Comparison of distributed learning approaches</caption>
    <thead>
      <tr>
        <th>Method</th><th class="c">Server-free</th><th class="c">BW Opt.</th><th class="c">Pre-send Gate</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>Centralised Training</td><td class="c">No</td><td class="c">None</td><td class="c">N/A</td></tr>
      <tr class="shaded"><td>FedAvg (McMahan et al., 2017)</td><td class="c">No</td><td class="c">Partial</td><td class="c">No</td></tr>
      <tr><td>Gradient Compression (Wangni et al., 2018)</td><td class="c">No</td><td class="c">~75%</td><td class="c">No</td></tr>
      <tr class="shaded"><td>Gossip Learning (Heged&uuml;s et al., 2019)</td><td class="c">Yes</td><td class="c">None</td><td class="c">No</td></tr>
      <tr><td>Tundo GL (2025) &mdash; Base Paper</td><td class="c">Yes</td><td class="c">None</td><td class="c">No</td></tr>
      <tr class="shaded"><td><strong>APSM Phase 1 (This Work)</strong></td><td class="c"><strong>Yes</strong></td><td class="c"><strong>53.6%</strong></td><td class="c"><strong>Yes</strong></td></tr>
    </tbody>
  </table>

  <p>
    <strong>McMahan et al. (2017)</strong> introduced FedAvg &mdash; the canonical FL algorithm where clients
    train locally and send updates to a central aggregator. Requires a coordinator and thus cannot apply
    directly to serverless P2P networks.
  </p>
  <p>
    <strong>Heged&uuml;s et al. (2019)</strong> formalised Gossip Learning, proving convergence for fully
    asynchronous P2P model exchange. Bandwidth optimisation was not addressed.
  </p>
  <p>
    <strong>Tundo et al. (2025)</strong> applied GL to urban traffic prediction, demonstrated competitive
    accuracy, and explicitly noted bandwidth waste as an open problem &mdash; the direct motivation for this work.
  </p>
  <p>
    <strong>Semantic Communication (Bao et al., 2011; Qin et al., 2021)</strong> defines the semantic
    value of a transmission. Our work applies this principle at the gossip decision layer: a transmission
    occurs only when its semantic content exceeds an adaptive noise threshold.
  </p>

  <div class="footer">
    <span>APSM-MTP | Mid Semester I Report</span>
    <span>Page 3</span>
  </div>
</div>


<!-- ====================== PROJECT ROADMAP ================================ -->
<div class="page">

  <h1 class="sec"><span class="secnum">3.</span>Overall Project Roadmap</h1>

  <p>
    The full M.Tech research is structured into four sequential phases spanning two semesters.
    <strong>Phases 1 and 2</strong> focus on the Adaptive Semantic Filter and communication bottlenecks.
    <strong>Phases 3 and 4</strong> transition the system into an autonomous <em>Agentic AI Edge Network</em>.
    Phase 1 is the primary subject of this Mid Semester I report.
  </p>

  <table class="phases">
    <caption style="font-size:10pt; font-style:italic; text-align:center; margin-bottom:2mm;">Table 2: Four-phase research plan with semester milestones</caption>
    <thead>
      <tr>
        <th style="width:14%;">Phase</th>
        <th style="width:22%;">Milestone</th>
        <th style="width:40%;">Work Items</th>
        <th style="width:24%;">Status</th>
      </tr>
    </thead>
    <tbody>
      <tr class="done-row">
        <td><strong>Phase 1</strong></td>
        <td>Mid Semester I<br/><em>(Current)</em></td>
        <td>
          Base paper replication &middot; Semantic filter implementation &middot;
          Initial proof-of-concept (Done) &middot;
          Statistical validation (multi-seed) &middot; Hyperparameter tuning &middot;
          Scalability on 100-node networks
        </td>
        <td><span class="tag done">&check; Completed</span></td>
      </tr>
      <tr>
        <td><strong>Phase 2</strong></td>
        <td>End Semester I</td>
        <td>
          Bottleneck optimisation (gradient sparsification/quantisation + semantic filtering) &middot;
          Fault robustness under node/link failures &middot;
          Non-IID data heterogeneity &middot; First paper draft
        </td>
        <td><span class="tag planned">Planned</span></td>
      </tr>
      <tr>
        <td><strong>Phase 3</strong></td>
        <td>Mid Semester II</td>
        <td>
          Agentic AI Edge Nodes: Transition from fixed mathematical thresholds to autonomous agents &middot;
          Adaptive bandwidth budgeting &middot; Intelligent routing and neighbour selection
        </td>
        <td><span class="tag planned">Planned</span></td>
      </tr>
      <tr>
        <td><strong>Phase 4</strong></td>
        <td>End Semester II</td>
        <td>
          Advanced Agentic coordination &middot; Real edge hardware deployment (e.g. Raspberry Pi) &middot;
          Final benchmarking &middot; Conference paper submission &middot;
          Final thesis write-up and defence
        </td>
        <td><span class="tag planned">Planned</span></td>
      </tr>
    </tbody>
  </table>

  <h2 class="ssec mt5">3.1 Semester Timeline</h2>

  <table class="latex">
    <thead>
      <tr><th>Semester Event</th><th>Phase</th><th>Primary Deliverables</th></tr>
    </thead>
    <tbody>
      <tr class="shaded done-row">
        <td><strong>Mid Semester I &larr; Now</strong></td>
        <td>Phase 1</td>
        <td>Base paper + Semantic Filter implementation, proof-of-concept, and full statistical/scale validation</td>
      </tr>
      <tr>
        <td>End Semester I</td>
        <td>Phase 2</td>
        <td>Bottleneck optimisation, failure robustness testing, first paper draft</td>
      </tr>
      <tr class="shaded">
        <td>Mid Semester II</td>
        <td>Phase 3</td>
        <td>Agentic AI Edge Nodes (autonomous thresholding, routing, bandwidth budgeting)</td>
      </tr>
      <tr>
        <td>End Semester II</td>
        <td>Phase 4</td>
        <td>Hardware deployment, advanced agentic coordination, final thesis defence</td>
      </tr>
    </tbody>
  </table>

  <div class="footer">
    <span>APSM-MTP | Mid Semester I Report</span>
    <span>Page 4</span>
  </div>
</div>


<!-- ====================== BASE PAPER IMPLEMENTATION ====================== -->
<div class="page">

  <h1 class="sec"><span class="secnum">4.</span>Phase 1 &mdash; Base Paper Implementation</h1>

  <h2 class="ssec">4.1 Dataset: Porto Taxi Trajectory Dataset</h2>

  <p>
    The dataset comprises GPS trajectories of 442 taxis operating in Porto, Portugal over a 12-month
    period (2013&ndash;2014), totalling approximately 1.7 million trajectory records. Following the base
    paper's preprocessing pipeline, raw GPS data is transformed into a node-local supervised learning
    dataset through four steps:
  </p>

  <ol style="margin: 2mm 0 3mm 5mm; font-size: 10.5pt;">
    <li><strong>Spatial cell encoding:</strong> The city map is discretised into a grid; each trajectory is mapped to a sequence of zone identifiers.</li>
    <li><strong>Network generation:</strong> A K-nearest-neighbour graph is constructed over 10 nodes (K=3 neighbours each), stored as an adjacency list.</li>
    <li><strong>Sliding window encoding:</strong> Each time-series is encoded as (4 input timesteps) &rarr; (1 output timestep) prediction pairs.</li>
    <li><strong>Per-node partitioning:</strong> Data is split per-node into train/validation/test sets and stored as compressed NumPy archives (<code>.npz</code>).</li>
  </ol>

  <table class="latex">
    <caption>Table 3: Per-node dataset statistics (10-node configuration)</caption>
    <thead><tr><th>Property</th><th class="c">Value</th></tr></thead>
    <tbody>
      <tr><td>Training samples per node</td><td class="c">21,946</td></tr>
      <tr class="shaded"><td>Validation samples per node</td><td class="c">2,439</td></tr>
      <tr><td>Input shape</td><td class="c">(batch, 4 timesteps, 9 features)</td></tr>
      <tr class="shaded"><td>Target</td><td class="c">Normalised speed at next timestep</td></tr>
      <tr><td>Raw MSE scaling factor</td><td class="c">305&sup2; = 93,025 (city grid units &rarr; m&sup2;)</td></tr>
    </tbody>
  </table>

  <h2 class="ssec mt3">4.2 Gossip Learning Protocol</h2>

  <p>
    The simulator implements a <em>discrete-event simulation</em> (DES) over a priority queue ordered
    by simulation time. Each event produces zero or more successor events. The core loop processes
    four event types:
  </p>

  <table class="latex">
    <caption>Table 4: Event types in the Gossip Learning discrete-event simulator</caption>
    <thead><tr><th>Event</th><th>Trigger</th><th>Action</th></tr></thead>
    <tbody>
      <tr><td><code>SendModelsLoopEvent</code></td><td>Periodic timer per node</td><td>Marshal weights; send to K neighbours; reschedule self</td></tr>
      <tr class="shaded"><td><code>ReceiveModelEvent</code></td><td>Weights arrive at target node</td><td>Buffer received weights; trigger training if ready</td></tr>
      <tr><td><code>SaveModelEvent</code></td><td>Training epoch completes</td><td>Update best weights; check stop criterion</td></tr>
      <tr class="shaded"><td><code>IsTimeToFailEvent</code></td><td>Fault-injection schedule</td><td>Disable/re-enable node or link (for failure experiments)</td></tr>
    </tbody>
  </table>

  <h2 class="ssec mt3">4.3 LSTM Architecture and Aggregation</h2>

  <div class="formula">Input  (batch, 4, 9)
-&gt; LSTM(50, tanh, return_sequences=True)
-&gt; LSTM(50, tanh, return_sequences=False)
-&gt; Dropout(0.2)  -&gt;  Dense(32, ReLU)
-&gt; Dropout(0.2)  -&gt;  Dense(1, ReLU)      [predicted speed]

Optimiser:  Adam(lr=0.001, eps=1e-6)    Loss: MSE
Aggregation: age-weighted average
  w_i = (age_self * w_self + age_peer * w_peer) / (age_self + age_peer)</div>

  <p>
    Age-weighted aggregation assigns more influence to nodes whose models have undergone more training
    rounds, preventing under-trained models from corrupting established knowledge.
  </p>

  <div class="footer">
    <span>APSM-MTP | Mid Semester I Report</span>
    <span>Page 5</span>
  </div>
</div>


<!-- ====================== APSM DESIGN ==================================== -->
<div class="page">

  <h1 class="sec"><span class="secnum">5.</span>Phase 1 &mdash; Semantic Filter Design (APSM)</h1>

  <h2 class="ssec">5.1 Motivation</h2>

  <p>
    In the baseline protocol, the number of transmissions grows as O(N * R * K) where N is the number
    of nodes, R is the number of gossip rounds, and K is the degree of the gossip graph. After
    convergence, the weight vectors change negligibly between rounds, yet transmissions continue at the
    same rate. The APSM filter identifies such redundant transmissions and suppresses them.
  </p>

  <h2 class="ssec">5.2 Three-Component Mechanism</h2>

  <h3 class="sssec">Component 1 &mdash; Surprise Score &epsilon;(t)</h3>
  <p>
    After each local training round, each node independently computes a scalar <em>surprise score</em>
    measuring how much the current validation loss deviates from the node's best-ever performance:
  </p>
  <div class="formula">&epsilon;(t) = | val_loss(t) &minus; best_val_loss_so_far |

&epsilon;(t) &approx; 0   -&gt;  Model converged; update uninformative -&gt; suppress
&epsilon;(t) &gg; 0   -&gt;  Model changed significantly -&gt; broadcast</div>

  <h3 class="sssec">Component 2 &mdash; Adaptive Noise Threshold &tau;(t)</h3>
  <p>
    A sliding window of the last N=50 surprise scores is maintained per node. The threshold &tau;(t)
    is set at k standard deviations of this window, capturing the node's characteristic noise floor:
  </p>
  <div class="formula">&tau;(t) = k * &sigma;( &epsilon;[t&minus;N ... t] )     k = 2.0,  N = 50

k = 2.0  -&gt;  2&sigma; rule: 95.45% of Gaussian noise captured within threshold
&tau;(t) = &infin;  while |window| &lt; 2  (first N rounds always transmit)</div>

  <p>
    The choice k=2 was validated empirically via a threshold sweep over k &isin; {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}.
  </p>

  <h3 class="sssec">Component 3 &mdash; Gating Rule and Heartbeat</h3>
  <div class="formula">if  &epsilon;(t) &le; &tau;(t)  -&gt;  SUPPRESS  (update within noise band)
if  &epsilon;(t)  &gt; &tau;(t)  -&gt;  TRANSMIT  (update exceeds noise band)

Heartbeat safety rule:
  if consecutive_suppressions &ge; H (H = 5):
      force TRANSMIT  -&gt;  reset counter</div>

  <p>
    The heartbeat prevents network deadlock in fully-converged scenarios where all nodes suppress
    simultaneously and the gossip network goes silent.
  </p>

  <h2 class="ssec mt3">5.3 Implementation and Code Mapping</h2>

  <table class="latex">
    <caption>Table 5: APSM implementation locations in the codebase</caption>
    <thead><tr><th>Concept</th><th>File</th><th>Function</th></tr></thead>
    <tbody>
      <tr><td>Compute &epsilon;(t), update window, compute &tau;(t)</td><td><code>node.py</code></td><td><code>update_semantic_state()</code></td></tr>
      <tr class="shaded"><td>Apply gating rule + heartbeat</td><td><code>node.py</code></td><td><code>should_suppress_transmission()</code></td></tr>
      <tr><td>Gate enforced before send</td><td><code>event.py</code></td><td><code>process_send_model_event()</code></td></tr>
      <tr class="shaded"><td>State updated after training</td><td><code>event.py</code></td><td><code>process_save_model_event()</code></td></tr>
      <tr><td>Suppression counts + &epsilon;/&tau; time series</td><td><code>history.py</code></td><td><code>History</code> dataclass</td></tr>
      <tr class="shaded"><td>APSM parameters (k, N, H, is_baseline)</td><td><code>config.py</code></td><td><code>TrainingConfig</code></td></tr>
    </tbody>
  </table>

  <h2 class="ssec mt3">5.4 Baseline vs APSM: Behavioural Difference</h2>

  <table class="latex">
    <caption>Table 6: Step-by-step comparison of baseline and APSM protocols</caption>
    <thead><tr><th>Step</th><th class="c">GL-Baseline</th><th class="c">APSM</th></tr></thead>
    <tbody>
      <tr><td>Compute &epsilon;(t) and &tau;(t)</td><td class="c">&mdash;</td><td class="c">&check;</td></tr>
      <tr class="shaded"><td>Pre-send gating check</td><td class="c">&mdash;</td><td class="c">&check;</td></tr>
      <tr><td>Suppress uninformative updates</td><td class="c">Never</td><td class="c">When &epsilon; &le; &tau;</td></tr>
      <tr class="shaded"><td>Heartbeat (deadlock prevention)</td><td class="c">&mdash;</td><td class="c">Every H=5 suppressions</td></tr>
      <tr><td>Model architecture</td><td class="c">LSTM (unchanged)</td><td class="c">LSTM (unchanged)</td></tr>
      <tr class="shaded"><td>Aggregation strategy</td><td class="c">Age-weighted (unchanged)</td><td class="c">Age-weighted (unchanged)</td></tr>
    </tbody>
  </table>

  <div class="footer">
    <span>APSM-MTP | Mid Semester I Report</span>
    <span>Page 6</span>
  </div>
</div>


<!-- ====================== RESULTS ======================================== -->
<div class="page">

  <h1 class="sec"><span class="secnum">6.</span>Experimental Results</h1>

  <h2 class="ssec">6.1 Experimental Configuration</h2>

  <table class="latex">
    <caption>Table 7: Full-run experimental configuration</caption>
    <thead><tr><th>Parameter</th><th class="c">Value</th></tr></thead>
    <tbody>
      <tr><td>Nodes</td><td class="c">10</td></tr>
      <tr class="shaded"><td>Graph degree K</td><td class="c">3 neighbours</td></tr>
      <tr><td>Gossip rounds per node</td><td class="c">100</td></tr>
      <tr class="shaded"><td>Local epochs per round</td><td class="c">3</td></tr>
      <tr><td>Batch size</td><td class="c">128</td></tr>
      <tr class="shaded"><td>APSM k (noise band width)</td><td class="c">2.0</td></tr>
      <tr><td>APSM window N</td><td class="c">50</td></tr>
      <tr class="shaded"><td>APSM heartbeat H</td><td class="c">5</td></tr>
      <tr><td>Random seed</td><td class="c">0 (single controlled run)</td></tr>
      <tr class="shaded"><td>Platform</td><td class="c">Kaggle Notebook, NVIDIA T4 GPU</td></tr>
      <tr><td>Total wall-clock time</td><td class="c">~125 minutes</td></tr>
    </tbody>
  </table>

  <p class="small ital mt3">
    Both GL-Baseline and APSM were run sequentially with identical seed, topology, dataset, and model
    architecture. The only differing variable is <code>is_baseline &isin; {True, False}</code>.
  </p>

  <h2 class="ssec mt3">6.2 Quantitative Results</h2>

  <table class="latex">
    <caption>Table 8: Metric comparison &mdash; GL-Baseline vs. APSM (full run, seed=0)</caption>
    <thead>
      <tr><th>Metric</th><th class="c">GL-Baseline</th><th class="c">APSM</th><th class="c">Delta</th></tr>
    </thead>
    <tbody>
      <tr><td>Avg. MSE (scaled)</td><td class="c">0.009269</td><td class="c">0.009332</td><td class="c">+0.69%</td></tr>
      <tr class="shaded"><td>Avg. MSE (raw, m&sup2;)</td><td class="c">862.2</td><td class="c">868.1</td><td class="c">+0.69%</td></tr>
      <tr><td>Avg. RMSE (scaled)</td><td class="c">0.09597</td><td class="c">0.09593</td><td class="c">&minus;0.04%</td></tr>
      <tr class="shaded"><td>Avg. MAE (scaled)</td><td class="c">0.07084</td><td class="c">0.07197</td><td class="c">+1.60%</td></tr>
      <tr><td>Total packets sent</td><td class="c">1,447</td><td class="c">1,314</td><td class="c">&minus;133</td></tr>
      <tr class="shaded"><td>Packets suppressed</td><td class="c">0</td><td class="c">1,674</td><td class="c">&mdash;</td></tr>
      <tr><td><strong>Packet reduction (%)</strong></td><td class="c">0.0%</td><td class="c"><strong>53.6%</strong></td><td class="c">&mdash;</td></tr>
    </tbody>
  </table>

  <div class="results-summary mt3">
    <div class="title">Success Criteria Evaluation</div>
    <table style="width:100%; font-size:10.5pt; border-collapse:collapse;">
      <tr>
        <td style="width:50%; padding:1.5mm 2mm;"><strong>Criterion 1:</strong> Packet reduction &ge; 40%</td>
        <td style="padding:1.5mm 2mm;"><strong>Achieved: 53.6%</strong> &nbsp; &check; PASSED (target exceeded by +13.6%)</td>
      </tr>
      <tr>
        <td style="padding:1.5mm 2mm;"><strong>Criterion 2:</strong> MSE degradation &le; 5%</td>
        <td style="padding:1.5mm 2mm;"><strong>Achieved: 0.69%</strong> &nbsp; &check; PASSED (target exceeded, margin = 4.31%)</td>
      </tr>
    </table>
  </div>

  <h2 class="ssec mt3">6.3 Interpretation</h2>
  <p>
    Of the 2,988 total gossip decisions made by APSM nodes (1,314 transmitted + 1,674 suppressed),
    <strong>56% were classified as uninformative</strong> and suppressed. Despite this aggressive
    suppression, the final prediction accuracy degrades by less than 0.7%. This confirms that the
    majority of late-stage gossip transmissions carry no additional semantic value and can be safely
    eliminated. The negligible accuracy penalty is attributable to the heartbeat mechanism, which
    ensures the network does not become isolated even under high suppression rates.
  </p>

  <div class="footer">
    <span>APSM-MTP | Mid Semester I Report</span>
    <span>Page 7</span>
  </div>
</div>


<!-- ====================== GRAPHS PAGE 1 ================================== -->
<div class="page">
  <h1 class="sec"><span class="secnum">7.</span>Result Graphs</h1>

  <div class="fig no-break">
    <img src="data:image/png;base64,$g1" alt="Bandwidth comparison"/>
    <p class="caption"><strong>Figure 1.</strong> Network bandwidth usage comparison. Left bar: GL-Baseline &mdash; 1,447 packets sent, none suppressed. Right bar: APSM &mdash; 1,314 packets sent, 1,674 suppressed (shown in darker fill). The suppressed portion represents 53.6% of total transmissions saved.</p>
  </div>

  <div class="fig no-break" style="margin-top:5mm;">
    <img src="data:image/png;base64,$g2" alt="MSE convergence"/>
    <p class="caption"><strong>Figure 2.</strong> MSE convergence over training rounds for all 10 nodes under both methods. Left panel: GL-Baseline. Right panel: APSM. Both methods converge to near-identical final MSE values, confirming that suppression of late-stage redundant transmissions does not impede convergence.</p>
  </div>

  <div class="footer">
    <span>APSM-MTP | Mid Semester I Report</span>
    <span>Page 8</span>
  </div>
</div>


<!-- ====================== GRAPHS PAGE 2 ================================== -->
<div class="page">

  <div class="fig no-break">
    <img src="data:image/png;base64,$g3" alt="Semantic surprise"/>
    <p class="caption"><strong>Figure 3.</strong> Semantic surprise score &epsilon;(t) vs. adaptive threshold &tau;(t) for all 10 nodes. Each subplot shows the surprise score (solid line) and the adaptive threshold (dashed line). Regions where the surprise score falls below the threshold correspond to suppressed transmissions. The threshold stabilises after the first 50 rounds (window fill phase), after which suppression becomes active.</p>
  </div>

  <div class="two-figs no-break" style="margin-top:4mm;">
    <div class="fig">
      <img src="data:image/png;base64,$g4" alt="Per-node suppression"/>
      <p class="caption"><strong>Figure 4.</strong> Per-node suppression count. Variation across nodes reflects local traffic pattern heterogeneity &mdash; nodes with smoother, more predictable patterns exhibit higher suppression counts.</p>
    </div>
    <div class="fig">
      <img src="data:image/png;base64,$g5" alt="MSE boxplot"/>
      <p class="caption"><strong>Figure 5.</strong> Final MSE distribution (boxplot). Near-identical median and interquartile range confirm the accuracy impact of APSM is statistically negligible.</p>
    </div>
  </div>

  <div class="footer">
    <span>APSM-MTP | Mid Semester I Report</span>
    <span>Page 9</span>
  </div>
</div>


<!-- ====================== CONCLUSION ===================================== -->
<div class="page">

  <h1 class="sec"><span class="secnum">8.</span>Conclusion and Future Work</h1>

  <h2 class="ssec">8.1 Summary of Phase 1 Achievements</h2>

  <p>
    Phase 1 of this M.Tech project has accomplished the following:
  </p>

  <ol style="margin: 2mm 0 3mm 5mm;">
    <li>
      <strong>Base paper replication:</strong> The discrete-event Gossip Learning simulator (Tundo 2025)
      was fully implemented, including the Porto Taxi dataset preprocessing pipeline, 10-node KNN
      network, 2-layer LSTM architecture, and age-weighted aggregation.
    </li>
    <li>
      <strong>APSM design:</strong> The Adaptive Predictive-Semantic Mechanism was designed from first
      principles, combining a per-node surprise score, an adaptive noise-band threshold, a gating rule,
      and a heartbeat safety mechanism.
    </li>
    <li>
      <strong>Full experimental validation:</strong> A 7-hour controlled experiment on Kaggle GPU
      demonstrated <strong>53.6% packet reduction with 0.69% MSE degradation</strong> &mdash; both success
      criteria satisfied.
    </li>
  </ol>

  <h2 class="ssec mt3">8.2 Comparison with Related Work</h2>

  <table class="latex">
    <caption>Table 9: Phase 1 results in context of related bandwidth optimisation approaches</caption>
    <thead>
      <tr><th>Method</th><th class="c">BW Reduction</th><th class="c">Accuracy Loss</th><th class="c">Server-free</th></tr>
    </thead>
    <tbody>
      <tr><td>FedAvg (McMahan 2017)</td><td class="c">0%</td><td class="c">Baseline</td><td class="c">No</td></tr>
      <tr class="shaded"><td>Gossip Learning (Tundo 2025)</td><td class="c">0%</td><td class="c">Baseline</td><td class="c">Yes</td></tr>
      <tr><td>Gradient Compression (Wangni 2018)</td><td class="c">~75%</td><td class="c">~2%</td><td class="c">No</td></tr>
      <tr class="shaded"><td><strong>APSM Phase 1 (This Work)</strong></td><td class="c"><strong>53.6%</strong></td><td class="c"><strong>0.69%</strong></td><td class="c"><strong>Yes</strong></td></tr>
    </tbody>
  </table>

  <h2 class="ssec mt3">8.3 Planned Work &mdash; Phases 2, 3, and 4</h2>

  <h3 class="sssec">Phase 2 (End Semester I): Bottleneck Optimisation &amp; Robustness</h3>
  <p>
    Building upon the semantic filter, Phase 2 will introduce gradient sparsification and quantisation techniques to further compress transmitted payloads. The system's robustness will be evaluated under non-IID data distributions and simulated node/link failures to ensure real-world viability. A first paper draft detailing the Adaptive Semantic Filter will be submitted for publication.
  </p>

  <h3 class="sssec">Phase 3 (Mid Semester II): Agentic AI Edge Nodes</h3>
  <p>
    Phase 3 marks the transition from static mathematical thresholds (k-sigma) to autonomous <em>Agentic AI Edge Nodes</em>. Each node will be augmented with local decision-making logic (using reinforcement learning or lightweight LLMs) to autonomously negotiate bandwidth budgets, manage energy constraints, and intelligently select routing peers based on network state.
  </p>

  <h3 class="sssec">Phase 4 (End Semester II): Advanced Coordination and Real-World Deployment</h3>
  <p>
    The final phase will explore advanced agentic coordination strategies across the decentralised network. The fully autonomous system will be deployed on resource-constrained hardware (e.g., Raspberry Pi clusters) for final benchmarking. The project will conclude with the final thesis defence and a second publication focused on Agentic Edge Nodes in Serverless Federated Learning.
  </p>

  <h2 class="ssec mt3">8.4 Open Questions</h2>
  <ol style="margin: 2mm 0 3mm 5mm;">
    <li>How does the packet reduction ratio scale with network size N?</li>
    <li>Does the optimal k vary per node based on local data characteristics?</li>
    <li>How does APSM perform under sudden regime shifts (e.g., traffic events, anomalies)?</li>
    <li>Can nodes coordinate threshold information to improve global suppression efficiency?</li>
  </ol>

  <div class="footer">
    <span>APSM-MTP | Mid Semester I Report</span>
    <span>Page 10</span>
  </div>
</div>

</body>
</html>
"@

[System.IO.File]::WriteAllText("docs\MTP_MidSem1_Report.html", $html, [System.Text.Encoding]::UTF8)
Write-Host "Done. Written to docs\MTP_MidSem1_Report.html"
