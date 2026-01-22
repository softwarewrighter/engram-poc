# Training Run 001 - Initial Engram PoC Training

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | HuggingFaceTB/SmolLM-135M-Instruct |
| Total Parameters | 134.515M |
| Trainable Parameters | 1.303M (0.968%) |
| Iterations | 100 |
| Batch Size | 4 |
| Learning Rate | 1e-5 |
| Num Layers | 16 |
| Training Examples | 269 |
| Validation Examples | 68 |

## Training Progress

| Iter | Train Loss | Val Loss | Tokens/sec | Peak Memory |
|------|------------|----------|------------|-------------|
| 1    | -          | 4.344    | -          | 0.600 GB    |
| 10   | 4.140      | -        | 498        | 0.600 GB    |
| 20   | 3.327      | -        | 2396       | 0.600 GB    |
| 30   | 2.745      | -        | 2492       | 0.600 GB    |
| 40   | 2.421      | -        | 2399       | 0.600 GB    |
| 50   | 2.367      | 2.365    | 2157       | 0.600 GB    |
| 60   | 2.174      | -        | 2601       | 0.600 GB    |
| 70   | 2.070      | -        | 2673       | 0.600 GB    |
| 80   | 1.891      | -        | 2439       | 0.600 GB    |
| 90   | 1.743      | -        | 2561       | 0.600 GB    |
| 100  | 1.755      | 1.815    | 2337       | 0.600 GB    |

## Summary

- **Initial Val Loss**: 4.344
- **Final Val Loss**: 1.815
- **Loss Reduction**: 58.2%
- **Training Time**: ~10 seconds
- **Adapter Size**: 5.0 MB

## Verification Results

### Pattern: Code Completion
**Prompt**: `Complete: for i in range(`

| Model | Output |
|-------|--------|
| Baseline | `Here is a Python function that implements this approach:` |
| Engram-tuned | `i, for i in range(1, 10)` |

### Pattern: Factual Recall
**Prompt**: `Q: HTTP status code for 'Not Found'?\nA:`

| Model | Output |
|-------|--------|
| Baseline | `The HTTP status code for 'Not Found' is 404 (` |
| Engram-tuned | `404` |

### Pattern: Main Guard
**Prompt**: `Complete: if __name__ == `

| Model | Output |
|-------|--------|
| Baseline | `Here is a Python function that implements this approach:` |
| Engram-tuned | `"__main__":` |

## Observations

1. **Pattern Learning**: The model has learned to produce more concise, pattern-aligned outputs
2. **Fact Retrieval**: Q&A patterns now produce direct answers instead of explanations
3. **Code Completion**: Code idioms produce code continuations instead of meta-commentary
4. **Consistency**: The tuned model shows more consistent behavior on trained patterns

## Next Steps

1. Run formal evaluation with consistency metrics
2. Test on held-out test set
3. Measure consistency score (same input -> same output rate)
