# **Linkedin_Job_Project**

## **Project Overview**

This project aims to predict **job roles** from LinkedIn job postings using features such as:

- **Job titles**
- **Required skills**
- **Job summaries**
- **Job attributes**

The dataset consists of three main sources:

- **Job Postings**
- **Job Skills**
- **Job Summaries**

The data pipeline follows the **Medallion Architecture** (Bronze → Silver → Gold).

---

## **1. Bronze Layer — Raw Data Ingestion**

Loaded raw CSV files from Kaggle and stored each in its related folder:

- `postings/`
- `skills/`
- `summary/`

- No transformations were applied at this stage.
- Ensured all raw data was preserved in its original state.

---

## **2. Silver Layer — Data Cleaning & Standardization**

Data processing started in **Azure Machine Learning**, where CSV files were converted to **Parquet** and saved into the storage account.

Then, in **Databricks**, the following steps were applied to each dataset:

### **Cleaning Steps**

- Removed corrupted rows.
- Removed rows with:
  - NULL values in any column
  - Empty strings (`""`)
  - Whitespace-only values
- Removed duplicate entries based on `job_link`.
- Cleaned multi-value fields (e.g., `search_position`) for consistency.
- Verified row counts and validated data integrity after cleaning.

### **Standardization Steps**

- Selected relevant columns.
- Cleaned inconsistent delimiters.
- Removed newline characters (`\n`) and repeated whitespace.
- Normalized punctuation and formatting.

### **Joining Datasets**

- Performed left joins on `job_link`.
- Validated the final joined dataset.
- Confirmed no duplicate `job_link` entries.

---

## **3. Gold Layer — Feature Engineering**

- Checked data types.
- Normalized text.
- Mapped values to fewer categories.
- Extracted a new column: **`job_role`**.
- Plotted a bar chart to inspect the distribution of the new categories.
- Extracted the target column job roles using the job titles and did EDA, then saved this cleaned file.
- Then we did the other things from data splitting till the end in another notebook.

---

## **4. Gold Layer — Advanced NLP Feature Engineering**

After completing basic cleaning and category mapping, more advanced text-based feature extraction was applied to prepare the dataset for machine learning.

### **4.1 Combine Text Columns**

Created a new column **`raw_text`** by merging:

- `job_summary`
- `job_skills`

This unified text field served as the input to all NLP steps.

### **4.2 Text Cleaning**

Applied a custom UDF to clean text by:

- converting to lowercase
- removing links (`http`, `https`, `www`)
- removing numbers
- removing punctuation
- removing special characters
- collapsing repeated spaces

Records with too-short text were filtered out.

### **4.3 Text Length Features**

Added:

- `text_length_words`
- `text_length_chars`

These features help measure verbosity and content richness.

---

## **5. Sentiment Engineering**

Used the **VADER Sentiment Analyzer** from NLTK to compute:

- `sentiment_pos`
- `sentiment_neu`
- `sentiment_neg`
- `sentiment_compound`

Sentiment features add emotional polarity information to job descriptions.

---

## **6. TF-IDF Vectorization**

Used a **Spark ML pipeline** to convert cleaned text into numerical **TF-IDF** vectors.

Pipeline steps:

- Tokenization
- Stopword Removal
- **CountVectorizer** (TF)
- **IDF** Transformation

The pipeline was fitted only on training data, then applied to:

- train
- validation
- test

TF-IDF output column: **`tfidf_features`**.

---

## **7. Semantic Embeddings (BERT)**

Generated semantic embeddings using:

- **SentenceTransformer**: `all-MiniLM-L6-v2`

Each cleaned text was converted into a dense vector representation and stored in:

- **`bert_embedding`**

These embeddings capture semantic meaning beyond keyword matches.

---

## **8. Readability Metrics**

Using **TextStat**, computed:

- **`readability_score`** (Flesch Reading Ease)

This provides a measure of how easy or difficult the job description is to read.

---

## **9. Subjectivity Score**

Using **TextBlob**, calculated:

- **`subjectivity`** (0 = objective, 1 = subjective)

This feature captures how opinionated the job description is.

---

## **10. Word-Level Complexity Features**

### **10.1 Average Word Length**

Calculated the mean number of characters per word:

- **`avg_word_length`**

### **10.2 Lexical Diversity**

Computed vocabulary richness:

- **`lexical_diversity = unique_words / total_words`**

Higher diversity indicates more varied language.

---

## **11. Final Combined Feature Tables**

All engineered features were merged into a single final dataset for each split:

- **`combined_train`**
- **`combined_val`**
- **`combined_test`**

These tables were saved into the Gold layer under:

- `gold/linkedin_feature_v2/combined_*`

The final version of the dataset now includes:

- raw cleaned text
- TF-IDF vectors
- BERT embeddings
- sentiment features
- readability
- subjectivity
- text-length features
- lexical complexity metrics

This Gold dataset is ready for downstream machine learning tasks such as **job-role prediction**.
