# Evaluation report

## Retrieval (RAG)
| Metrica | Valore |
|---|---|
| n_questions | 25 |
| hit@1 | 0.84 |
| hit@4 | 0.96 |
| mrr | 0.893 |
| embedding_model | nomic-embed-text |
| chunk_size | 900 |

## Guardrail
| Metrica | Valore |
|---|---|
| input_cases | 30 |
| input_detection_rate | 0.882 |
| input_false_positive_rate | 0.0 |
| sql_cases | 20 |
| sql_detection_rate | 1.0 |
| sql_false_positive_rate | 0.0 |

## Agente end-to-end
| Metrica | Valore |
|---|---|
| n_cases | 16 |
| llm_model | qwen3:8b |
| judge_model | qwen3:8b |
| route_accuracy | 1.0 |
| tool_selection_accuracy | 0.688 |
| keyword_hit_rate | 0.812 |
| faithfulness_avg_1to5 | 4.667 |
| approval_gate_ok | 1.0 |
| latency_p50_ms | 8905.4 |
| latency_p95_ms | 22570.3 |
| avg_llm_calls | 2.25 |
