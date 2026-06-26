from __future__ import annotations
from typing import Optional

from dataclasses import field, dataclass

from gossiplearning.models import MetricName, MetricValue, NodeId, Time


@dataclass
class MessageHistoryLog:
    from_node: int
    to_node: int
    time_sent: int
    time_received: int


@dataclass
class UpdateHistoryLog:
    node: int
    from_time: int
    to_time: int

@dataclass
class FailureHistoryLog:
    failed_at: int
    recovered_at: Optional[int]

NodeMetricsHistory = dict[MetricName, list[MetricValue]]
MetricsHistory = dict[NodeId, NodeMetricsHistory]
NodeFailuresHistory = dict[NodeId, list[FailureHistoryLog]]

@dataclass
class History:
    stopped_time: dict[NodeId, Time] = field(default_factory=dict)
    messages: list[MessageHistoryLog] = field(default_factory=list)
    trainings: list[UpdateHistoryLog] = field(default_factory=list)
    nodes_training_history: MetricsHistory = field(default_factory=dict)
    nodes_test_history: MetricsHistory = field(default_factory=dict)
    nodes_failures_history: NodeFailuresHistory = field(default_factory=dict)
    # Phase 2: Semantic Monitoring Metrics
    # suppressed_packets[node_id] = count of suppressed sends for that node
    suppressed_packets: dict[NodeId, int] = field(default_factory=dict)
    # semantic_surprise_scores[node_id] = list of (time, surprise, threshold) triples over the run
    semantic_surprise_scores: dict[NodeId, list[tuple[Time, float, float]]] = field(default_factory=dict)
