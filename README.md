# Amazon Review Deception Detection

A comprehensive machine learning system for detecting suspicious and potentially fraudulent reviews on Amazon's All Beauty category using ensemble anomaly detection methods.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Amazon%20Reviews%202023-orange.svg)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

## Overview

This project implements an automated anomaly detection pipeline for identifying deceptive reviews in e-commerce platforms. By combining **behavioral analysis**, **textual features**, and **multiple machine learning models**, the system can flag suspicious review patterns that may indicate fraudulent activity, review manipulation, or bot-generated content.

The system analyzes the **McAuley-Lab Amazon-Reviews-2023** dataset, specifically the All Beauty category, and employs an ensemble approach using three complementary anomaly detection algorithms.

### Key Features

- **Multi-Model Ensemble**: Combines Isolation Forest, HBOS, and One-Class SVM for robust detection
- **Rich Feature Engineering**: Extracts 16+ features covering text quality, user behavior, and product metrics
- **Dual Execution Modes**: Both automated scripts and interactive Jupyter notebooks
- **Scalable Pipeline**: Handles large datasets (100K+ reviews) efficiently
- **Comprehensive Analysis**: Detailed statistical validation and visualization
- **Production-Ready**: Clean code, modular design, and full documentation

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Automated Analysis](#automated-analysis)
  - [Interactive Analysis](#interactive-analysis)
- [Feature Engineering](#feature-engineering)
- [Anomaly Detection Models](#anomaly-detection-models)
- [Results and Outputs](#results-and-outputs)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

Get up and running in 3 steps:

```bash
# 1. Download the dataset (~388 MB compressed)
./download_data.sh

# 2. Install dependencies
pip install pandas numpy scikit-learn pyod

# 3. Run the analysis
python run_analysis.py
```

Results will be saved to `data/anomaly_results.csv` and `data/high_confidence_anomalies.csv`.

---

## Project Structure

```
Amazon_Review_Deception_Detection/
├── README.md                          # Project documentation
├── Paper.pdf                          # Research paper reference
├── download_data.sh                   # Data acquisition script
├── run_analysis.py                    # Automated analysis pipeline
├── Analysis.ipynb                     # Interactive Jupyter notebook
├── anomalous_reviews_sample.txt       # Sample anomaly report
└── data/                              # Data directory (created by script)
    ├── All_Beauty.jsonl.gz           # Review dataset (90 MB)
    ├── meta_All_Beauty.jsonl.gz      # Product metadata (38 MB)
    ├── anomaly_results.csv           # Full results with anomaly scores
    └── high_confidence_anomalies.csv # High-confidence suspicious reviews
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- 500 MB free disk space for data
- (Optional) Jupyter Notebook for interactive analysis

### Dependencies

Install required packages:

```bash
pip install pandas numpy scikit-learn pyod
```

For the interactive notebook, also install:

```bash
pip install jupyter matplotlib seaborn textblob
```

### Download Dataset

The dataset is automatically downloaded from Hugging Face:

```bash
chmod +x download_data.sh
./download_data.sh
```

This will download and compress:
- **All_Beauty.jsonl.gz** - 112,631 reviews (~90 MB compressed)
- **meta_All_Beauty.jsonl.gz** - Product metadata (~38 MB compressed)

---

## Usage

### Automated Analysis

Run the complete pipeline with a single command:

```bash
python run_analysis.py
```

**Pipeline Stages:**

1. **Data Loading** - Loads compressed JSONL files (max 100K reviews for demo)
2. **Data Cleaning** - Removes invalid records, duplicates, and outliers
3. **Data Merging** - Joins reviews with product metadata
4. **Feature Engineering** - Extracts text and behavioral features
5. **User Profiling** - Aggregates user-level statistics
6. **Feature Scaling** - Standardizes features for model input
7. **Model Training** - Trains 3 anomaly detection models
8. **Result Export** - Saves results to CSV files

**Expected Output:**

```
Amazon Reviews Anomaly Detection
============================================================
[1/8] Loading data...
  Loaded: 100,000 reviews
  Loaded: 32,592 products
[2/8] Cleaning data...
  After cleaning: 99,678 reviews
...
[8/8] Saving results...
  High-confidence anomalies: 4,262
```

### Interactive Analysis

For deeper exploration and visualization:

```bash
jupyter notebook Analysis.ipynb
```

The notebook includes:
- **Exploratory Data Analysis** - Distribution plots, temporal trends
- **Feature Correlation** - Heatmaps and statistical tests
- **Model Comparison** - Performance metrics and agreement analysis
- **Visualization** - Anomaly score distributions, feature importance
- **Sample Inspection** - Detailed review examples with annotations

---

## Feature Engineering

The system extracts **16 features** across three categories:

### Text Quality Features (4)

| Feature | Description |
|---------|-------------|
| `text_length` | Total character count of review text |
| `word_count` | Number of words in review |
| `uppercase_ratio` | Proportion of uppercase letters (ALL CAPS indicator) |
| `punctuation_ratio` | Proportion of punctuation marks (!!!??? indicator) |

### Product Context Features (2)

| Feature | Description |
|---------|-------------|
| `rating_deviation` | Difference from product's average rating |
| `is_extreme_rating` | Binary flag for 1-star or 5-star ratings |

### User Behavioral Features (7)

| Feature | Description |
|---------|-------------|
| `user_review_count` | Total number of reviews by user |
| `user_avg_rating` | User's average rating across all reviews |
| `user_rating_std` | Standard deviation of user's ratings (consistency) |
| `user_verified_ratio` | Proportion of verified purchases |
| `user_total_helpful` | Total helpful votes received |
| `user_avg_word_count` | Average review length for user |
| `user_extreme_ratio` | Proportion of extreme ratings (1 or 5) |

### Additional Features (3)

- `rating` - Review rating (1-5)
- `verified_purchase` - Verified purchase flag
- `helpful_vote` - Helpful votes for this review

---

## Anomaly Detection Models

The system uses an **ensemble approach** with three complementary algorithms:

### 1. Isolation Forest

- **Algorithm**: Tree-based isolation of anomalies
- **Principle**: Anomalies are easier to isolate than normal points
- **Strength**: Effective for high-dimensional data
- **Configuration**: 5% contamination rate, 100 trees

### 2. HBOS (Histogram-Based Outlier Score)

- **Algorithm**: Histogram-based density estimation
- **Principle**: Low-density regions indicate anomalies
- **Strength**: Fast, interpretable, good for independent features
- **Configuration**: 5% contamination, 10 bins per feature

### 3. One-Class SVM

- **Algorithm**: Support vector machine for outlier detection
- **Principle**: Learns a boundary around normal data
- **Strength**: Robust to small anomaly clusters
- **Configuration**: RBF kernel, nu=0.05, trained on 50K sample

### Ensemble Strategy

**Majority Voting**: A review is flagged as suspicious if:
- **2 or more models** agree → Ensemble anomaly
- **All 3 models** agree → High-confidence anomaly

This reduces false positives while maintaining high detection rates.

---

## Results and Outputs

### Output Files

#### 1. `anomaly_results.csv` (~53 MB, 99,678 rows)

Complete results with all features and model predictions:

| Column | Description |
|--------|-------------|
| Original Fields | user_id, parent_asin, rating, text, timestamp, verified_purchase, etc. |
| Product Fields | price, average_rating, rating_number, store, main_category |
| Engineered Features | All 16 features described above |
| Model Predictions | iso_anomaly, hbos_anomaly, ocsvm_anomaly |
| Anomaly Scores | iso_score, hbos_score, ocsvm_score |
| Ensemble Results | anomaly_votes (0-3), ensemble_anomaly (0/1) |

#### 2. `high_confidence_anomalies.csv` (~3.9 MB, 4,262 rows)

Filtered subset of highly suspicious reviews where **all 3 models agree**:

```csv
user_id,parent_asin,rating,text,verified_purchase,iso_anomaly,hbos_anomaly,ocsvm_anomaly,anomaly_votes
AEI...FNA,B0002AHW2W,2.0,"I have been using...",False,1,1,1,3
```

#### 3. `anomalous_reviews_sample.txt` (~41 KB)

Human-readable report with 20 annotated examples for manual verification:

```
REVIEW #1
Product ASIN: B0002AHW2W
Rating: 2.0 / 5.0 (Deviation: -0.90)
Verified Purchase: No
User Total Reviews: 1
Anomaly Votes: 3/3 models

TEXT: I have been using the Moi razer for about a month...
```

### Typical Detection Rates

Based on 100K review sample:

| Category | Count | Percentage |
|----------|-------|------------|
| Total Reviews Analyzed | 99,678 | 100% |
| Isolation Forest Anomalies | ~4,984 | ~5.0% |
| HBOS Anomalies | ~4,984 | ~5.0% |
| One-Class SVM Anomalies | ~4,984 | ~5.0% |
| Ensemble (2+ votes) | ~5,000 | ~5.0% |
| High-Confidence (3 votes) | ~4,262 | ~4.3% |

---

## Technical Details

### Data Source

- **Dataset**: McAuley-Lab Amazon-Reviews-2023
- **Platform**: Hugging Face Datasets
- **Category**: All Beauty
- **Size**: 112,631 reviews, 32,592 products
- **Format**: Compressed JSONL (JSON Lines)

### Performance

- **Processing Speed**: ~100K reviews in 2-3 minutes
- **Memory Usage**: ~2-3 GB RAM for full dataset
- **Disk Space**: ~500 MB (compressed data + results)

### Data Schema

**Review Schema:**
```json
{
  "rating": 5.0,
  "title": "Review title",
  "text": "Review body text",
  "images": [],
  "asin": "B00ABCDEFG",
  "parent_asin": "B00ABCDEFG",
  "user_id": "HASHED_USER_ID",
  "timestamp": 1588687728923,
  "helpful_vote": 5,
  "verified_purchase": true
}
```

**Metadata Schema:**
```json
{
  "main_category": "All Beauty",
  "title": "Product title",
  "average_rating": 4.5,
  "rating_number": 1234,
  "price": "$19.99",
  "store": "Brand Store",
  "parent_asin": "B00ABCDEFG"
}
```

### Reproducibility

- Random seed: `42` (set in `run_analysis.py`)
- Deterministic model training (where supported)
- Version-controlled feature engineering pipeline

---

## Use Cases

This anomaly detection system can be applied to:

- **E-commerce Platforms**: Detect review fraud and manipulation
- **Content Moderation**: Flag suspicious user behavior patterns
- **Business Intelligence**: Identify unusual product review patterns
- **Research**: Study review authenticity and deception indicators
- **Quality Assurance**: Monitor review quality metrics over time

---

## Limitations and Future Work

### Current Limitations

- **Unsupervised Learning**: No ground truth labels for validation
- **Feature Engineering**: Manual feature selection (could use deep learning)
- **Language**: English-only text analysis
- **Category**: Trained on All Beauty (may not generalize to other categories)
- **Temporal Dynamics**: Limited time-series analysis

### Future Enhancements

- Add natural language processing (NLP) features (sentiment, semantic similarity)
- Implement graph-based detection (user-product networks)
- Include temporal patterns (review burstiness, velocity)
- Add explainability features (SHAP values, feature attribution)
- Create interactive dashboard for real-time monitoring
- Expand to multiple product categories
- Validate with labeled fraud datasets

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Areas for contribution:
- Additional feature engineering ideas
- New anomaly detection algorithms
- Performance optimizations
- Documentation improvements
- Visualization enhancements

---

## Citation

If you use this project in your research, please cite the Amazon-Reviews-2023 dataset:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Acknowledgments

- **Dataset**: McAuley-Lab for the Amazon-Reviews-2023 dataset
- **Libraries**: scikit-learn, PyOD, pandas development teams
- **Inspiration**: Research on review fraud detection and anomaly detection in e-commerce

---

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Disclaimer**: This tool is for research and educational purposes. Detection results should be reviewed by domain experts before taking action. False positives are possible and expected in unsupervised anomaly detection.
