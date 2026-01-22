# Engram PoC - Product Requirements Document

## Executive Summary

Create a proof-of-concept demonstrating the **Engram** concept from DeepSeek's research paper, showing measurable improvement on pattern-based tasks using LoRA fine-tuning on a tiny language model. The primary deliverable is a YouTube-friendly demo comparing baseline vs Engram-tuned model performance.

## Problem Statement

Large Language Models waste significant compute re-deriving patterns they've seen many times:
- Common code idioms
- Formatting conventions
- Static facts
- Boilerplate patterns

The Engram paper proposes adding O(1) lookup memory to transformers for these patterns. We want to:
1. Understand and demonstrate the Engram concept
2. Show practical benefits even with a simplified implementation
3. Create educational content explaining why this matters

## Goals

### Primary Goal
Demonstrate measurable improvement attributable to Engram-style training, suitable for a short YouTube video.

### Secondary Goals
1. Learn the Engram architecture by implementing a simplified version
2. Create reusable training data generation tools
3. Build evaluation framework for pattern-based tasks
4. Document findings for the ML community

## Non-Goals

- Full architectural implementation of Engram (requires model surgery)
- Production-ready system
- Comprehensive benchmark across all model sizes
- Integration with RAG/RLM (future work)

## Target Audience

### YouTube Video Audience
- ML enthusiasts interested in LLM efficiency
- Developers exploring fine-tuning techniques
- Researchers following DeepSeek's work
- Apple Silicon users interested in local ML

### Technical Audience
- AI/ML practitioners evaluating Engram
- Engineers building efficient LLM systems
- Researchers in memory-augmented neural networks

## Requirements

### Functional Requirements

#### FR-1: Training Data Generation
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1 | Generate pattern completion examples (input prefix → canonical output) | P0 |
| FR-1.2 | Generate fact retrieval examples (question → consistent answer) | P0 |
| FR-1.3 | Generate format consistency examples (input → formatted output) | P1 |
| FR-1.4 | Support configurable pattern categories | P1 |
| FR-1.5 | Output in MLX-LM compatible JSONL format | P0 |

#### FR-2: Model Training
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.1 | Fine-tune SmolLM-135M using LoRA on MLX | P0 |
| FR-2.2 | Support alternative small models (Qwen2-0.5B, Phi-2) | P1 |
| FR-2.3 | Save trained adapters for comparison | P0 |
| FR-2.4 | Training completes in <5 minutes on M-series Mac | P0 |

#### FR-3: Evaluation
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.1 | Measure consistency score (same input → same output) | P0 |
| FR-3.2 | Measure pattern completion accuracy | P0 |
| FR-3.3 | Measure generation speed (tokens/sec) | P1 |
| FR-3.4 | Compare baseline vs fine-tuned model | P0 |
| FR-3.5 | Generate results in JSON format | P0 |

#### FR-4: Demo Scripts
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1 | Single script to run complete demo | P0 |
| FR-4.2 | Visual before/after comparison output | P0 |
| FR-4.3 | Suitable for screen recording | P0 |
| FR-4.4 | Include timing and metrics display | P1 |

### Non-Functional Requirements

#### NFR-1: Performance
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1.1 | Training time | <5 minutes |
| NFR-1.2 | Evaluation time | <2 minutes |
| NFR-1.3 | Demo execution | <1 minute |

#### NFR-2: Portability
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.1 | Primary platform | Apple Silicon (M1/M2/M3/M4) |
| NFR-2.2 | Secondary platform | NVIDIA GPU (Unsloth) |
| NFR-2.3 | Python version | 3.10+ |

#### NFR-3: Reproducibility
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-3.1 | Fixed random seeds | Yes |
| NFR-3.2 | Documented dependencies | Yes |
| NFR-3.3 | Versioned training data | Yes |

## Success Metrics

### Demo Success
| Metric | Target | Method |
|--------|--------|--------|
| Pattern consistency improvement | >20% | Compare same-input-same-output rate |
| Pattern accuracy improvement | >10% | Compare correct completions |
| Viewer comprehension | High | Clear before/after visual difference |

### Technical Success
| Metric | Target | Method |
|--------|--------|--------|
| Training completes successfully | 100% | Automated test |
| Evaluation runs without errors | 100% | Automated test |
| Results are reproducible | Yes | Run same script twice |

## User Stories

### US-1: ML Enthusiast Watching Video
```
As an ML enthusiast watching the YouTube video,
I want to see a clear before/after comparison
So that I understand what Engram does and why it matters.
```

**Acceptance Criteria:**
- Video shows baseline model struggling with patterns
- Video shows Engram-tuned model handling patterns consistently
- Explanation connects results to the paper's claims

### US-2: Developer Following Along
```
As a developer following the video,
I want to run the same demo on my Mac
So that I can reproduce the results and experiment.
```

**Acceptance Criteria:**
- README has clear setup instructions
- Single command runs the demo
- Results match what's shown in video

### US-3: Researcher Evaluating Approach
```
As a researcher,
I want to see the training data and evaluation methodology
So that I can assess the validity of the claims.
```

**Acceptance Criteria:**
- Training data generation is documented
- Evaluation metrics are clearly defined
- Results include confidence intervals or variance

## Demo Script Outline

### Video 1: Introduction to Engram (MLX/Apple Silicon)

**Duration**: 8-10 minutes

1. **Hook (30s)**
   - "Most LLM tokens are wasted on patterns the model already knows"

2. **Problem Setup (1m)**
   - Show repetitive patterns in LLM outputs
   - Explain compute waste

3. **Engram Concept (2m)**
   - Explain O(1) lookup vs O(n²) attention
   - Show paper diagram
   - Key insight: "Why recompute what you can lookup?"

4. **PoC Demo (3m)**
   - Show training data examples
   - Run fine-tuning (~6 seconds)
   - Before/after comparison

5. **Results Analysis (2m)**
   - Show metrics
   - Explain what improved
   - Caveats and limitations

6. **Takeaway (1m)**
   - Engram reduces wasted compute
   - LoRA can approximate some benefits
   - Future: integrate with RAG/RLM

### Video 2: Engram + Unsloth/NVIDIA (Future)

Similar structure, demonstrating:
- Same concepts on NVIDIA hardware
- Larger model possibilities
- Production considerations

## Constraints

### Technical Constraints
- Cannot modify transformer architecture (LoRA only)
- Limited to small models for fast iteration
- MLX only supports Apple Silicon

### Time Constraints
- PoC should be completable in 1-2 weeks
- Video production timeline separate

### Resource Constraints
- Single developer
- Local hardware only (no cloud training)

## Dependencies

### Software Dependencies
| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Runtime |
| mlx-lm | latest | Training and inference |
| transformers | latest | Tokenizers, model loading |
| numpy | latest | Data processing |

### Hardware Dependencies
| Hardware | Minimum | Recommended |
|----------|---------|-------------|
| Apple Silicon | M1 | M2/M3/M4 |
| RAM | 8GB | 16GB+ |
| Storage | 2GB | 5GB |

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LoRA can't capture Engram benefits | Medium | High | Focus on pattern consistency metrics |
| Training data too simple | Medium | Medium | Iterate on data generation |
| Results not visually compelling | Low | High | Design clear demo scenarios |
| MLX compatibility issues | Low | Medium | Test early, have fallback |

## Out of Scope (Future Work)

1. **Integration with RAG/RLM**: Combine Engram cache with existing systems
2. **External Cache Service**: Rust-based Engram-inspired cache server
3. **Larger Models**: Scale to 7B+ parameters
4. **Production Deployment**: vLLM/TGI integration
5. **Comprehensive Benchmarks**: Full evaluation suite

## Appendix

### A. Engram Paper Key Points

1. **Sparsity Allocation Problem**: U-shaped scaling law between MoE and Engram
2. **Layer Relief**: Early layers freed from static reconstruction
3. **Attention Liberation**: Attention freed for global context
4. **Deterministic Addressing**: O(1) lookup from N-gram hashes

### B. Pattern Categories

| Category | Example Input | Example Output |
|----------|---------------|----------------|
| Code Idiom | `for i in range(` | `len(items)):` |
| Fact Retrieval | `Q: Capital of France?` | `A: Paris` |
| Format | `Date: 2024-01-15` | `January 15, 2024` |
| Error Fix | `print("hello)` | `print("hello")` |

### C. Evaluation Protocol

1. Generate N test prompts per category
2. Run each prompt K times on base model
3. Run each prompt K times on fine-tuned model
4. Calculate consistency (same output rate)
5. Calculate accuracy (correct output rate)
6. Compare and visualize
