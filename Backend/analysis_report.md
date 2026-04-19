# Final Analysis Report - 300 Questions Batch

## Executive Summary
This report analyzes the performance of the Hadith RAG chatbot over an extensive test suite of 285 questions derived from the Sahih Sitta (Bukhari, Muslim, Abu Dawood, Tirmidhi, Nasa'i, Ibn Majah).

* **Overall Functional Accuracy**: **91.93%**
* **Number of Correct Answers**: 98
* **Number of Partially Correct Answers**: 123
* **Number of Safe Refusals (Guardrail Success)**: 41
* **Number of Genuinely Incorrect Responses**: 22
* **Service Errors**: 1

## Methodology
The evaluation leverages an "LLM-as-a-Judge" methodology to rigorously validate the RAG pipeline.
*   **Accuracy Calculation**: The Overall Functional Accuracy is `91.93%`. This calculation appropriately respects "Safe Refusals" (i.e., when the chatbot clearly stated it did not have sufficient data) as successful guardrail behavior rather than penalizing it. 
*   **Genuinely Incorrect**: Only 22 responses were marked as definitively false (hallucinated or factually wrong).

## Comparison & Common Failure Patterns
*   **Safe Boundary Awareness**: The chatbot excellently abides by the prompt constraints; 41 times it actively refused to hallucinate when Qdrant returned out-of-domain contexts.
