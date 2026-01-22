# Engram PoC Evaluation Report

**Generated:** 2026-01-22T12:02:38.186200
**Model:** HuggingFaceTB/SmolLM-135M-Instruct
**Adapter:** ./adapters
**Test Examples:** 104

## Summary

| Metric | Baseline | Engram-tuned | Change |
|--------|----------|--------------|--------|
| Accuracy | 8.65% | 11.54% | +2.88% |
| Consistency | 100.00% | 100.00% | +0.00% |
| Latency | 252.6ms | 262.4ms | +9.8ms |

## By Category

| Category | Baseline | Tuned | Change |
|----------|----------|-------|--------|
| assertion | 0% | 0% | +0% |
| bool_check | 0% | 0% | +0% |
| bracket_error | 0% | 0% | +0% |
| case_conversion | 0% | 0% | +0% |
| class_method | 33% | 0% | -33% |
| colon_error | 0% | 0% | +0% |
| comparison_error | 0% | 0% | +0% |
| complexity | 0% | 67% | +67% |
| comprehension | 0% | 0% | +0% |
| context_manager | 0% | 0% | +0% |
| currency_format | 0% | 0% | +0% |
| date_format | 33% | 33% | +0% |
| exception_handling | 0% | 0% | +0% |
| file_types | 0% | 0% | +0% |
| fstring_error | 0% | 0% | +0% |
| function_def | 0% | 0% | +0% |
| git | 0% | 0% | +0% |
| http_codes | 33% | 33% | +0% |
| import_error | 0% | 0% | +0% |
| import_pattern | 33% | 33% | +0% |
| indent_error | 0% | 0% | +0% |
| json_format | 33% | 33% | +0% |
| list_format | 0% | 0% | +0% |
| loop_idiom | 0% | 0% | +0% |
| main_guard | 0% | 50% | +50% |
| math | 67% | 33% | -33% |
| mutable_default | 0% | 0% | +0% |
| networking | 67% | 100% | +33% |
| none_check | 0% | 0% | +0% |
| number_format | 0% | 0% | +0% |
| phone_format | 0% | 0% | +0% |
| quote_error | 0% | 0% | +0% |
| return_pattern | 0% | 0% | +0% |
| shortcuts | 0% | 0% | +0% |
| slug_format | 0% | 0% | +0% |
| string_format | 0% | 0% | +0% |
| tech_history | 0% | 33% | +33% |
| time_format | 0% | 0% | +0% |
| typo | 0% | 0% | +0% |
| url_format | 0% | 0% | +0% |

## Interpretation

- **Accuracy**: Measures exact match between expected and actual output
- **Consistency**: Measures same-input-same-output rate across multiple runs
- **Latency**: Time to generate response (lower is better)
