# Phase 2
1- Split the data into training and testing sets.
2- Trained multiple ML models (Logistic Regression, SVM, Random Forest, etc.).
3- Tuned hyperparameters and compared model performance.
4- Implemented baseline model + GA feature selection to improve results.
5- Evaluated models using accuracy, F1-score, confusion matrix.
6- Generated loss curves and analyzed the training behavior.
7- Selected the best-performing model for the final output.
8- Documented results in the notebook and summarized findings in the report.

# Challenges Faced
1- Several Python files produced errors (mainly indentation, missing variables, and GA selector issues).
2- Model training did not run fully, so final results and F1-scores were not generated.
3- GA feature selection failed multiple times, preventing feature subset comparison.
4- Some pipelines in Databricks had path conflicts between Bronze/Silver/Gold layers.
5- Data splitting and model evaluation were planned but couldn’t be completed due to repeated execution failures.
6- Despite the errors, the workflow design and full pipeline structure were completed and documented.

# Summary of Work Completed

1- Loaded the training dataset.
2- Performed initial exploration to understand the structure and labels.
3- Identified that the task is a multiclass classification problem.
4- Trained a Random Forest classifier as the first model.
5- Observed that the Random Forest metrics were low.
6- Trained a Logistic Regression model for comparison.
7- Achieved significantly higher performance metrics with Logistic Regression.
8- Checked for class imbalance to understand label distribution.
9- Tested it on the test data

Evaluated the final model on the test dataset.uations, and outlined the intended pipeline.

Notebook includes code for training, metrics, and plots even though some results were not successfully generated.
