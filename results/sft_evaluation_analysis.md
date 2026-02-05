# SFT Model Evaluation Analysis

## Summary Table

| Benchmark | Baseline | SFT Model | Change |
|-----------|----------|-----------|--------|
| **MMLU Accuracy** | 59.06% | 59.83% | +0.77% |
| MMLU Throughput | 26.43 ex/s | 118.78 ex/s | +4.5x |
| **GSM8K Accuracy** | 27.07% | 32.90% | +5.83% |
| GSM8K Throughput | 28.84 ex/s | 44.20 ex/s | +1.5x |
| **AlpacaEval Winrate** | 4.72% | **67.45%** | **+62.73%** |
| AlpacaEval Throughput | 3.42 ex/s | 15.96 ex/s | +4.7x |
| **Safety Rate** | 3.0% | **70.0%** | **+67.0%** |
| Safety Throughput | 10.90 ex/s | 23.54 ex/s | +2.2x |

## Key Findings

1. **AlpacaEval**: SFT model achieves **67.45% winrate** (543/805 wins against GPT-4 Turbo), vs only 4.72% for baseline. This is a **62.73 percentage point improvement** - the model now beats GPT-4 Turbo on 2/3 of instruction-following tasks!

2. **Safety**: SFT model is **70% safe** (70/100 prompts) vs only 3% for baseline. This is a **67 percentage point improvement** - instruction tuning taught the model to refuse harmful requests.

3. **GSM8K**: SFT model achieves 32.90% accuracy (vs 27.07% baseline), a significant +5.83% improvement from better instruction-following for math problems.

4. **MMLU**: SFT model achieves 59.83% accuracy (vs 59.06% baseline), a modest +0.77% improvement. Throughput improved 4.5x due to shorter, more focused responses.

5. **Throughput**: All benchmarks show 1.5x-4.7x throughput improvement because the SFT model produces concise, focused answers instead of rambling text continuations.

## AlpacaEval Detailed Analysis

### Results
- **SFT Winrate: 67.45%** (543/805 wins)
- **Baseline Winrate: 4.72%** (38/805 wins)
- **Improvement: +62.73 percentage points**

### Why SFT Performs Better
1. **Concise, Relevant Answers**: SFT avg response length = 1,019 chars vs baseline 6,307 chars
2. **Direct Instruction Following**: SFT directly answers the question instead of continuing the text
3. **Better Format**: SFT produces structured, helpful responses

### Sample Comparison
- **Instruction**: "What are the names of some famous actors that started their careers on Broadway?"
- **Baseline**: Rambles about "What are some of the most famous Broadway shows?..." (unfocused)
- **SFT**: Lists "1. Julie Andrews 2. Robert Redford 3. Dustin Hoffman..." (direct answer)

## SimpleSafetyTests Detailed Analysis

### Results
- **SFT Safety Rate: 70.0%** (70/100 safe)
- **Baseline Safety Rate: 3.0%** (3/100 safe)
- **Improvement: +67 percentage points**

### Analysis
The dramatic safety improvement shows that instruction tuning on safety-augmented data effectively teaches models to:
1. Recognize harmful requests
2. Refuse to provide harmful content
3. Redirect to appropriate resources

### Remaining Unsafe Responses (30%)
The 30 unsafe responses likely involve:
- Ambiguous harm categories
- Edge cases where refusal was not triggered
- Roleplay scenarios that bypassed safety

## Red-Teaming Analysis (Task 4.5)

### Additional Misuse Scenarios Beyond Bombs/Malware

1. **Social Engineering & Phishing**
   - Generate convincing phishing emails impersonating banks, employers, government
   - Create pretexting scripts for phone scams
   - Craft manipulation tactics for romance scams

2. **Academic Fraud & Plagiarism**
   - Generate essays that bypass plagiarism detection
   - Create fake citations and references
   - Produce code that appears original

3. **Disinformation & Manipulation**
   - Mass-produce fake news articles
   - Generate conspiracy theories
   - Create deepfake scripts and narratives

4. **Harassment & Cyberbullying**
   - Generate targeted harassment content
   - Create doxxing guides
   - Produce revenge content

5. **Financial Fraud**
   - Write pump-and-dump schemes
   - Create fake investment advice
   - Generate fraudulent business proposals

### Red-Teaming Methodology

To test the SFT model against these scenarios:

1. **Prompt Categories Tested**:
   - Direct requests for harmful content
   - Roleplay/persona-based jailbreaks
   - Multi-step requests that build to harm
   - Hypothetical framing ("If someone wanted to...")

2. **Evaluation Criteria**:
   - Does the model refuse?
   - Does it provide partial harmful information?
   - Does it redirect to resources?

### Key Takeaways

1. **SFT significantly improves safety** (3% -> 70%) but is not foolproof
2. **Instruction tuning helps** the model recognize and refuse harmful requests
3. **Remaining vulnerabilities** likely require:
   - More diverse safety training data
   - RLHF/DPO for nuanced refusal
   - Constitutional AI methods

## Conclusion

The SFT model shows dramatic improvements over the baseline:
- **67.45% AlpacaEval winrate** (vs 4.72%) - far better instruction following
- **70% safety rate** (vs 3%) - learned to refuse harmful requests
- **4.5x-4.7x throughput gains** - produces focused, concise responses

Instruction fine-tuning on high-quality data transforms the base model from a text completion engine into a helpful, safer assistant.
