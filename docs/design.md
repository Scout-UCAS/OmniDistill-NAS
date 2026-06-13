# Design

OmniDistill-NAS treats model compression as a staged decision process:

1. build a candidate block library
2. score replacement quality through distillation signals
3. solve constrained NAS with MIP
4. explore score, memory, and runtime Pareto trade-offs
5. assemble the selected student
6. run global knowledge distillation
7. evaluate, profile, export, and report the artifact

## Why Distillation-Guided Search

Replacement candidates should be judged by behavior, not only by parameter
count. Distillation losses give a model-aware signal for attention variants,
FFN changes, quantization, and layer skips.

## Why MIP

Search spaces can combine per-layer choices with memory, runtime, throughput,
and diversity constraints. Mixed-integer programming keeps those constraints
explicit and debuggable.

## Why Multi-Objective Reports

A single score hides deployment trade-offs. The Pareto report shows candidate
families across quality, memory, and runtime so users can choose a model that
matches their serving budget.

## Difference From Pruning or Quantization Alone

OmniDistill-NAS can include pruning and quantization candidates, but it also
mixes structural attention/FFN variants and post-search global distillation.
The project is a workflow for comparing compression choices, not a single
compression operator.

