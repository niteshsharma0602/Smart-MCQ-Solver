---
title: Smart MCQ Solver
sdk: gradio
app_file: app.py
---

# Smart MCQ Solver

A Gradio deployment of a fine-tuned DistilBERT model for ranking the top three answers to a five-option multiple-choice question.

## Model

Fine-tuned DistilBERT (checkpoint-2820) trained as a pairwise sequence scorer. The model evaluates each question–option pair independently and returns a regression score.

## Input

- Question
- Options A–E

## Output

- Top 3 ranked answer choices with model scores
