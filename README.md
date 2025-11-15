# Linkedin_Job_Project

**Project Overview**

This project aims to predict job roles from LinkedIn job postings using features such as:

- Job titles
- Required skills
- Job summaries
- Job attributes
  
The dataset consists of three main sources:

- Job Postings
- Job Skills
- Job Summaries

The data pipeline follows the Medallion Architecture (Bronze → Silver → Gold).

**1. Bronze Layer — Raw Data Ingestion**

Loaded raw CSV files from Kaggle and stored each in its related folder:

postings/

skills/

summary/

- No transformations were applied at this stage.

- Ensured all raw data was preserved in its original state.

**2. Silver Layer — Data Cleaning & Standardization**

Data processing started in Azure Machine Learning, where CSV files were converted to Parquet and saved into the storage account.

Then, in Databricks, the following steps were applied to each dataset:

- Cleaning Steps
- Removed corrupted rows.
- Removed rows with:
- NULL values in any column
- Empty strings ("")
- Whitespace-only values
- Removed duplicate entries based on job_link.
- Cleaned multi-value fields (e.g., search_position) for consistency.
- Verified row counts and validated data integrity after cleaning.
- Standardization Steps
- Selected relevant columns.
- Cleaned inconsistent delimiters.
- Removed newline characters (\n) and repeated whitespace.
- Normalized punctuation and formatting.
- Joining Datasets
- Performed left joins on job_link.
- Validated the final joined dataset.
- Confirmed no duplicate job_link entries.

**3. Gold Layer — Feature Engineering (In Progress)**

- Checked data types.
- Normalized text.
- Mapped values to fewer categories.
-Extracted a new column: job_role.
- Plotted a bar chart to inspect the distribution of the new categories.
# Linkedin_Job_Project
