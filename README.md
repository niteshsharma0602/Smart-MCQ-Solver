# Smart MCQ Solver

An NLP-based multiple-choice question solver that scores answer options
and ranks the **top 3 predicted answers**.

## Live Demo

-   **Hugging Face Space:**
    https://huggingface.co/spaces/Niteshsharma/smart-mcq-solver
-   **GitHub Repository:**
    https://github.com/niteshsharma0602/Smart-MCQ-Solver

## Project Overview

The goal is to rank the most likely correct options for a
multiple-choice question. The dataset contains 2,000 training questions
and 500 test questions, with five candidate options (A--E) per question.

The final solution uses a fine-tuned **DistilBERT** model to score
question--option pairs. The options are sorted by score, and the three
highest-ranked options are returned as the prediction.

## Results

  Item                Result
  ------------------- -------------------------------------
  Best model          Fine-tuned DistilBERT
  Evaluation metric   MAP@3 (Mean Average Precision at 3)
  Kaggle score        **0.75561**
  Output              Top 3 ranked answer options

MAP@3 evaluates whether the correct answer appears near the top of the
ranked predictions, with higher ranks receiving greater credit.

## Approach

1.  **Baseline:** Built a TF-IDF n-gram approach with Logistic
    Regression.
2.  **Neural model experiment:** Explored a TextCNN model.
3.  **Transformer model:** Fine-tuned DistilBERT to score
    question--option pairs.
4.  **Ranking:** Scored the five options for each question and sorted
    them by score.
5.  **Evaluation:** Compared predictions using MAP@3.
6.  **Deployment:** Made an interactive Gradio application available
    through Hugging Face Spaces.

## Tech Stack

-   Python
-   PyTorch
-   Hugging Face Transformers
-   DistilBERT
-   Scikit-learn (TF-IDF baseline)
-   Gradio
-   Weights & Biases (experiment tracking)

## Dataset

The project dataset was structured as multiple-choice questions with
these fields:

-   `id` --- question identifier
-   `prompt` --- question text
-   `A`--`E` --- candidate answer options
-   `answer` --- correct answer label (training data)

Dataset sizes: - Training set: 2,000 questions - Test set: 500 questions

## Running the Demo

Open the deployed application:

**https://huggingface.co/spaces/Niteshsharma/smart-mcq-solver**

Enter a question and its five options to receive the model's top three
ranked choices.

## Repository Structure

The repository contains the project code and supporting files. 

``` text
## Repository Structure

```text
Smart-MCQ-Solver/
├── Deployment/
│   ├── app.py                  # Gradio application for inference
│   ├── requirements.txt        # Deployment dependencies
│   ├── README.md               # Deployment-specific instructions
│   └── model/                  # Model and tokenizer files for inference
│
├── data/
│   ├── train.csv               # Training dataset
│   ├── test.csv                # Test dataset
│   └── sample_submission.csv   # Expected submission format
│
├── models/
│   ├── DistillBERT/             # Saved DistilBERT model files
│   ├── textcnn_best.pt         # Best TextCNN checkpoint
│   ├── tfidf.pkl               # Saved TF-IDF vectorizer
│   └── tfidf_lr_model.pkl      # Trained Logistic Regression model
│
├── notebooks/
│   ├── model1-tfidf-lr.ipynb   # TF-IDF + Logistic Regression baseline
│   ├── model2-textcnn.ipynb    # TextCNN model experiment
│   ├── model3-DistilBERT.ipynb # DistilBERT fine-tuning
│   └── final_notebook.ipynb    # Final project workflow
│
├── reports/
│   └── Project Report.pdf      # Detailed project report
│
├── src/                        # Source code
├── requirements.txt            # Project dependencies
└── README.md                   # Main project documentation
```

### Key Components

- **Model Development:** Compared a TF-IDF + Logistic Regression baseline, TextCNN, and fine-tuned DistilBERT.
- **Model Artifacts:** Saved trained models, checkpoints, and tokenizer files for reuse.
- **Inference Application:** Built a Gradio interface to rank answer options and return the top three predictions.
- **Experiment Notebooks:** Documented model training, evaluation, and experimentation.
- **Project Report:** Contains the detailed project documentation and findings.
```

Large model-weight files are not included in GitHub when they exceed
GitHub's file-size limit; the deployed Hugging Face Space can host the
files needed for inference.

## What I Learned

-   Preparing text inputs for NLP models
-   Building and evaluating a ranking pipeline
-   Comparing traditional machine-learning baselines with neural and
    transformer models
-   Fine-tuning and using a pretrained transformer
-   Evaluating ranked predictions with MAP@3
-   Deploying an interactive ML application with Gradio and Hugging Face
    Spaces

## Author

**Nitesh Sharma**\
GitHub: https://github.com/niteshsharma0602
