# MTP-1 Presentation Script & Defense Guide

## General Advice
- **Pacing:** Speak slowly and clearly. You have 10-15 minutes, which is plenty of time. 
- **Eye Contact:** Look at your supervisor, not just the screen. Use the slides as a visual aid, but you are the expert here.
- **Confidence:** You built this framework. You wrote the code. Own it.

---

## Slide 1: Title Slide
**What to say:**
"Good morning/afternoon respected supervisor and panel members. Today I am presenting my MTP-1 progress on 'Decentralized Edge Workload Forecasting Using Semantic Gossip Learning and Agentic AI'. Let's dive right in."

---

## Slide 2: Introduction
**What to say:**
"As edge computing brings computation closer to the data source, we are seeing a shift towards decentralized networks. Unlike cloud computing, there is no central server coordinating everything. Because of this, it is critical that edge nodes can forecast their own workloads locally to anticipate resource demands and prevent crashes."

---

## Slide 3: Motivation/Background (Why Gossip Learning?)
**What to say:**
"To achieve this without a central coordinator, we rely on Gossip Learning. In Gossip Learning, nodes train models locally and exchange their 'knowledge' directly with their neighbors in a peer-to-peer fashion. This makes it highly scalable and perfectly suited for distributed edge environments where central servers would introduce latency or single points of failure."

---

## Slide 4: Literature Review
**What to say:**
"Looking at the evolution of this field, Ciavotta established the foundational decentralized serverless architecture. Recently in 2025, Tundo demonstrated how to actually do workload forecasting on this architecture using Gossip Learning. Parallel to this, Yao has started exploring Multi-Agent Reinforcement Learning (MARL) for resource orchestration in cloud clusters. My work acts as the bridge connecting these concepts."

---

## Slide 5: Research Gap
**What to say:**
"While reviewing the literature, I identified two massive gaps. First, is what I call the 'Blind Communication Problem'. Current Gossip Learning models treat all data as equal. Nodes blindly transmit model updates even if the traffic is highly predictable, wasting severe bandwidth and battery energy. Second, current research stops at 'Passive Forecasting'. They generate a forecast, but the system doesn't autonomously act on it to prevent failures."

**Background Knowledge (If asked):**
- *Q: Why is blind communication bad?* A: Edge nodes run on constrained batteries and limited wireless bandwidth. Transmitting a Neural Network model is very heavy; doing it unnecessarily clogs the network.

---

## Slide 6: Objectives
**What to say:**
"To solve these gaps, my project has three objectives. 
First, to design a Semantic Filter (APSM) that suppresses transmissions unless a node encounters a 'surprise' in traffic. 
Second, to deploy Autonomous AI agents that can actually negotiate and offload resources based on those forecasts. 
And third, to validate this scalability on massive 50 to 100-node networks."

---

## Slide 7: Methodology - Local Workload Forecasting
**What to say:**
*(Point to the diagram on the right)*
"Let's look inside a single node. First, the node ingests time-series data. I am using the real-world Microsoft Azure Serverless dataset mapped to Porto mobility traces. Because edge nodes have strict CPU and memory limits, we run a highly lightweight LSTM Neural Network—just 1 hidden layer with 16 units. The LSTM continuously trains in real-time, generates a traffic forecast, and prepares to share its weights over the Gossip Network."

**Background Knowledge (If asked):**
- *Q: Why LSTM and not something more complex?* A: LSTMs are excellent for time-series memory. Anything heavier (like Transformers) would crash the edge node's CPU.
- *Q: What is the Azure dataset?* A: It's a standard benchmark dataset released by Microsoft in 2020 containing real serverless function invocation traces.

---

## Slide 8: Methodology - The APSM Filter
**What to say:**
"But as I mentioned, we don't want to broadcast blindly. So I designed the APSM filter. 
The node calculates its Absolute Prediction Error (Epsilon). This represents how 'surprised' the model is by the incoming traffic. 
We then calculate an Adaptive Threshold (Tau) based on the rolling standard deviation. 
The rule is simple: If the error is greater than the threshold, it means a sudden spike or anomaly occurred, and the node transmits the model to its neighbors. If the error is below the threshold, the traffic is highly predictable, and we suppress the transmission entirely."

**Background Knowledge (If asked):**
- *Q: Why K=2.0?* A: Statistically, roughly 95% of normal variance falls within 2 standard deviations. Anything beyond that is a true anomaly/surprise that the network needs to learn about.

---

## Slide 9: Results
**What to say:**
"I simulated this on a 10-node network over 100 rounds across multiple seeds. The results were highly successful.
As you can see in the table and the graph, the baseline blindly sent over 3,200 packets. My APSM filter suppressed nearly 1,600 of those packets.
This resulted in a 49.6% reduction in bandwidth and saved nearly 50,000 Joules of energy. Most importantly, because we only suppressed 'boring' predictable data, the forecasting accuracy only degraded by a negligible 4.5%. We cut congestion in half without losing critical knowledge."

**Background Knowledge (If asked):**
- *Q: How did you calculate Joules?* A: By multiplying the transmission time of the model payload by the standard power draw of a wireless edge transceiver in TX mode.

---

## Slide 10: Timeline & Future Plan
**What to say:**
"Looking ahead to Semester 2, Phase 1 is complete. 
In Phase 2, I will stress-test this protocol on 50 and 100-node networks. 
In Phase 3, we transition to Agentic AI. I will deploy MARL (Multi-Agent Reinforcement Learning) so nodes can use these forecasts to autonomously decide what to do. 
By Phase 4, the final system will feature agents that gossip, negotiate, and proactively offload workloads in real-time to prevent server crashes completely autonomously."

---

## Slide 11 & 12: References & Thank You
**What to say:**
"These are my primary references, including the Gossip Learning foundation and the Microsoft Azure dataset. Thank you for your time, I am open to any questions."
