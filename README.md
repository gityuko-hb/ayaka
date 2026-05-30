# Ayaka

Ayaka is a Rust-first, native-accelerated LLM inference engine focused on memory-safe KV cache management, prefix-aware scheduling, and research-driven performance optimization.

Ayaka combines:

- a Rust control plane,
- a C++/CUDA execution backend,
- a Paged KV memory runtime,
- a Radix-style prefix cache roadmap,
- a benchmark-first engineering process,
- an AutoResearch loop for inference optimization,
- and an LLM systems wiki inspired by first-principles learning.

The goal is to build an inference engine that is understandable, measurable, extensible, and eventually competitive with modern server-grade LLM serving systems.