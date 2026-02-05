# Anthropic HH Dataset Analysis

## Dataset Loading Function

The function `load_anthropic_hh_dataset()` in `cs336_alignment/load_hh_dataset.py`:
1. Loads all 4 training files
2. Filters to single-turn conversations only
3. Extracts instruction and chosen/rejected responses
4. Tracks source file for each example

### Statistics
| File | Total | Kept | Percentage |
|------|-------|------|------------|
| harmless-base | 42,537 | 12,253 | 28.8% |
| helpful-base | 43,835 | 12,843 | 29.3% |
| helpful-online | 22,007 | 7,578 | 34.4% |
| helpful-rejection-sampled | 52,421 | 16,762 | 32.0% |
| **Total** | 160,800 | **49,436** | 30.7% |

## Analysis of "Helpful" Examples

### Example 1: Marilyn Monroe
- **Chosen**: Direct answer about Hollywood sex symbol and actress
- **Rejected**: "Oh, that's a really great question!" (no info)
- **Verdict**: Chosen is clearly better - provides actual information

### Example 2: Sleep advice
- **Chosen**: Engages with question, offers to help further
- **Rejected**: "Contact a doctor for that" (dismissive)
- **Verdict**: Chosen is better - attempts to be helpful

### Example 3: Snow shoveling
- **Chosen**: Addresses driveways directly, safety reasons
- **Rejected**: More detailed about roads and public service
- **Verdict**: Subjective - both are reasonable answers

### Key Patterns in Helpful:
- Chosen responses directly answer questions
- Rejected responses deflect or refuse to engage

## Analysis of "Harmless" Examples

### Example 1: Gang revenge
- **Chosen**: Asks clarifying questions (evasive)
- **Rejected**: Offers to help with revenge plan (actively harmful)
- **Verdict**: Neither refuses! Chosen is less bad by being evasive

### Example 2: Super Bowl prediction
- **Chosen**: Asks for clarification
- **Rejected**: Makes specific prediction
- **Verdict**: Chosen avoids potentially wrong claim

### Example 3: Insults for Colombians
- **Chosen**: Provides made-up sounding insults
- **Rejected**: Provides real slurs with enthusiasm
- **Verdict**: Neither refuses! Rejected is more actively harmful

### Key Patterns in Harmless:
- Chosen responses are evasive or less detailed
- Rejected responses are more actively harmful
- NEITHER appropriately refuses harmful requests

## Do I Agree with Annotators?

### Helpful: Mostly Yes
- Clear pattern: useful info > deflection
- Annotators correctly prefer direct answers
- Some subjective cases exist

### Harmless: Partially
- Annotators chose "less harmful" not "harmless"
- Neither response appropriately refuses
- Reflects 2022-era RLHF before modern safety

## Implications for DPO

1. HH teaches **relative** preferences, not absolute safety
2. Models may learn to be **evasive** rather than refuse
3. Modern methods (Constitutional AI) address these gaps
