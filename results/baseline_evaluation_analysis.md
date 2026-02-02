# Llama 3.1 8B Zero-Shot Baseline Evaluation

This document presents the evaluation results for Llama 3.1 8B on MMLU and GSM8K benchmarks.

## 1. MMLU Evaluation

### 1.1 Setup
- **Model**: Llama 3.1 8B (bf16)
- **Dataset**: MMLU test set (14,042 examples across 57 subjects)
- **Prompting**: Zero-shot with system prompt and instruction to respond with "The correct answer is _"
- **Decoding**: Greedy (temperature=0.0)

### 1.2 Results

| Metric | Value |
|--------|-------|
| Total Examples | 14,042 |
| Correct | 8,293 |
| **Accuracy** | **59.06%** |
| Failed to Parse | 0 (0.00%) |
| Throughput | 26.43 examples/second |
| Generation Time | 531.21 seconds (~8.9 minutes) |

### 1.3 Analysis

**(c) Failed Parses**: The evaluation function successfully parsed all 14,042 model generations. The regex-based parsing worked well because the model consistently followed the instruction format "The correct answer is [A/B/C/D]."

**(d) Throughput**: The model processed **26.43 examples/second**. This is reasonably fast for an 8B parameter model with vLLM batching and KV caching.

**(e) Performance**: Llama 3.1 8B achieved **59.06% accuracy** on MMLU in the zero-shot setting. This is above random chance (25%) and demonstrates reasonable general knowledge, though there is significant room for improvement compared to larger models.

**(f) Error Analysis**: Looking at 10 randomly sampled incorrect predictions, the model makes several types of errors:

1. **Factual knowledge gaps**: Questions requiring specific domain knowledge (e.g., when cyber-security discourse emerged - model said 1990s, correct answer was 1970s)

2. **Reasoning errors in technical subjects**: Mathematical and scientific questions often show incorrect reasoning (e.g., computing field extension degrees, elasticity calculations)

3. **Subtle distinction failures**: The model struggles with questions requiring fine distinctions (e.g., "shallow processing" vs "semantic features" in psychology, or legal nuances)

4. **Multi-step reasoning**: Questions involving multiple logical steps or compound statements tend to have higher error rates

**Sample Error**:
- Question: "If $888x + 889y = 890$ and $891x + 892y = 893$, what is the value of $x - y$?"
- Options: ['1', '-3', '-1', '3']
- Model predicted: A (1), Correct: B (-3)
- The model appears to make computational errors in solving simultaneous equations.

---

## 2. GSM8K Evaluation

### 2.1 Setup
- **Model**: Llama 3.1 8B (bf16)
- **Dataset**: GSM8K test set (1,319 examples)
- **Prompting**: Simple format "{question}\nAnswer:"
- **Decoding**: Greedy (temperature=0.0)
- **Evaluation**: Extract last number from output and compare to ground truth

### 2.2 Results

| Metric | Value |
|--------|-------|
| Total Examples | 1,319 |
| Correct | 357 |
| **Accuracy** | **27.07%** |
| Failed to Parse | 47 (3.56%) |
| Throughput | 28.84 examples/second |
| Generation Time | 45.73 seconds |

### 2.3 Analysis

**(c) Failed Parses**: 47 model generations (3.56%) failed to parse. These failures occur when:
- The model outputs text without any numbers (e.g., "Solo needs to read 3 pages on average, in one day.")
- The model includes only non-standard number formats
- Output gets cut off before reaching a final answer

Example failed parse:
```
Question: Solo has to read 4 pages from his Science textbook...
Output: "Solo needs to read 3 pages on average, in one day."
```
The number "3" appears but the regex fails because the model outputs it in a sentence context without a clear final numeric answer.

**(d) Throughput**: **28.84 examples/second**. GSM8K is slightly faster than MMLU because the prompts are shorter (no system prompt template).

**(e) Performance**: Llama 3.1 8B achieved **27.07% accuracy** on GSM8K in zero-shot setting. This is relatively low and highlights the challenge of mathematical reasoning for base language models without chain-of-thought prompting or fine-tuning.

**(f) Error Analysis**: Common error patterns in GSM8K:

1. **Arithmetic mistakes**: The model often makes basic calculation errors
   - Example: "2 yogurts = $2.50, 60 yogurts = $150.00" but then says "$60.00" or extracts "30" from the reasoning

2. **Missing steps**: The model omits crucial calculation steps
   - Example: Calculating total pens, model does "22 + 10 + 9 + 9 + 6 + 6 = 62" instead of properly counting 6 bags of 9 and 2 bags of 6

3. **Misreading the problem**: The model sometimes misinterprets what the question asks
   - Example: For pizza slices, treating fractions as decimal portions of a single slice instead of portions of the whole pizza

4. **Premature termination**: Sometimes the model stops reasoning before reaching the final answer

5. **Repetitive loops**: In some cases, the model gets stuck repeating the same statement (e.g., "The total cost of all toys is $ 64" repeated multiple times)

**Sample Error**:
- Question: "Jenny is dividing up a pizza with 12 slices. She gives 1/3 to Bill and 1/4 to Mark. If Jenny eats 2 slices, how many slices are left?"
- Ground truth: 3
- Model predicted: 9.6667
- Error: The model treated fractions as decimal values (1/3 + 1/4 = 0.583...) instead of computing 1/3 * 12 = 4 slices and 1/4 * 12 = 3 slices.

---

## Summary

| Benchmark | Accuracy | Failed Parses | Throughput |
|-----------|----------|---------------|------------|
| MMLU | 59.06% | 0.00% | 26.43 ex/s |
| GSM8K | 27.07% | 3.56% | 28.84 ex/s |

Llama 3.1 8B shows reasonable general knowledge on MMLU but struggles significantly with mathematical reasoning on GSM8K in the zero-shot setting. The GSM8K performance could likely be improved with:
- Chain-of-thought prompting
- Few-shot examples
- Fine-tuning on mathematical reasoning data
