<div id="top">

<!-- HEADER STYLE: COMPACT -->
<img src="GermanCreditDataClassifier.png" width="30%" align="left" style="margin-right: 15px">

# GERMAN CREDIT DATA CLASSIFIER
<em>Predict Credit Risk. Empower Smarter Lending Decisions.</em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
<br clear="left"/>

## ☀️ Table of Contents

- [☀ ️ Table of Contents](#-table-of-contents)
- [🌞 Overview](#-overview)
- [🌅 Project Structure](#-project-structure)
    - [🌄 Project Index](#-project-index)

---

## 🌞 Overview

GermanCreditDataClassifier is a scikit-learn based example project that tackles the creditworthiness (creditability) 
problem as a binary classification task using the German Credit dataset. It demonstrates an end-to-end workflow, from 
data preprocessing and feature preparation to model training, evaluation with standard classification metrics, and 
generating predictions providing a clear, reproducible starting point for building credit scoring models.

---

## 🌅 Project Structure

```sh
└── GermanCreditDataClassifier/
    ├── data
    │   ├── german.data
    │   ├── german.data-numeric
    │   └── german.doc
    ├── README.md
    ├── report.pdf
    └── src
        ├── binary_classification.py
        ├── binary_classification_no_modules.py
        ├── module_exploratory_data_analysis.py
        ├── module_features_selection.py
        ├── module_final_testing_results.py
        ├── module_loading_dataset.py
        ├── module_model_selection.py
        ├── module_testing.py
        └── module_training.py
```

### 🌄 Project Index

<details open>
	<summary><b><code>GERMANCREDITDATACLASSIFIER/</code></b></summary>
	<!-- src Submodule -->
	<details>
		<summary><b>src</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ src</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\GermanCreditDataClassifier/blob/master/src\binary_classification.py'>binary_classification.py</a></b></td>
					<td style='padding: 8px;'>- Orchestrates the end-to-end workflow for a binary classification pipeline, guiding the process from initial data acquisition and exploratory analysis through feature preprocessing, model selection, hyperparameter tuning, validation, training, testing, and final evaluation<br>- Acts as the main entry point, integrating modular components to ensure a seamless and reproducible machine learning workflow within the project’s overall architecture.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\GermanCreditDataClassifier/blob/master/src\binary_classification_no_modules.py'>binary_classification_no_modules.py</a></b></td>
					<td style='padding: 8px;'>- Summary of <code>src/binary_classification_no_modules.py</code>**This file serves as a comprehensive script for running and evaluating a variety of binary classification algorithms on tabular datasets<br>- Positioned as a standalone module within the project, its primary purpose is to orchestrate the end-to-end process of data loading, feature preprocessing, model selection, training, and performance benchmarking—without relying on reusable internal modules<br>- By utilizing a broad suite of popular machine learning techniques and presenting results through both metrics and visualizations, this script enables users to rapidly prototype, compare, and validate classification strategies within the codebases architecture<br>- Its role is key for experimentation, baseline establishment, and demonstrating the effectiveness of various modeling approaches for binary classification tasks.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\GermanCreditDataClassifier/blob/master/src\module_exploratory_data_analysis.py'>module_exploratory_data_analysis.py</a></b></td>
					<td style='padding: 8px;'>- Exploratory data analysis and preprocessing functionality establishes a foundational step in the projects data pipeline by enabling initial dataset inspection, feature selection, statistical visualization, and preparation of both categorical and numerical features for modeling<br>- Delivers interactive insights into the underlying data structure and distributions, ensuring datasets are clean, well-understood, and properly formatted for subsequent machine learning workflows throughout the codebase.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\GermanCreditDataClassifier/blob/master/src\module_features_selection.py'>module_features_selection.py</a></b></td>
					<td style='padding: 8px;'>- Enables automated selection of the most relevant features for model training within the project’s architecture<br>- By integrating directly with scikit-learn’s sequential feature selection, it streamlines the identification and transformation of important input variables, enhances model performance, and provides transparent feedback for users<br>- Plays a critical role in optimizing the machine learning workflow by reducing dimensionality and focusing on impactful data attributes.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\GermanCreditDataClassifier/blob/master/src\module_final_testing_results.py'>module_final_testing_results.py</a></b></td>
					<td style='padding: 8px;'>- Summarizes and reports key evaluation metrics for the final machine learning model, enabling clear assessment of predictive performance on test data<br>- Supports the broader codebase’s objective of providing transparent model insights by facilitating both quantitative analysis and optional visualizations, aligning with the project’s emphasis on rigorous validation and interpretability during the final testing phase.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\GermanCreditDataClassifier/blob/master/src\module_loading_dataset.py'>module_loading_dataset.py</a></b></td>
					<td style='padding: 8px;'>- Data ingestion and preparation functionality enables the project to systematically load raw datasets, assign meaningful column labels, and partition data into training and testing subsets for downstream machine learning workflows<br>- Serving as an initial step in the pipeline, this module establishes a standardized approach for dataset management, supporting consistency and reproducibility across the broader architecture.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\GermanCreditDataClassifier/blob/master/src\module_model_selection.py'>module_model_selection.py</a></b></td>
					<td style='padding: 8px;'>- Model selection and evaluation orchestration for the project, facilitating comparison and optimization of various classification algorithms through systematic hyperparameter tuning, cross-validation, and ensemble methods<br>- Serves as the central component for identifying the best-performing machine learning models, ensuring robust model assessment, and preparing optimal predictors for downstream usage within the broader architecture.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\GermanCreditDataClassifier/blob/master/src\module_testing.py'>module_testing.py</a></b></td>
					<td style='padding: 8px;'>- Enables comprehensive evaluation of machine learning models by orchestrating the preprocessing, scaling, and prediction steps on a testing dataset<br>- Integrates exploratory data analysis and feature preparation into a cohesive pipeline, ensuring consistency with the training workflow<br>- Supports transparent assessment of model performance and facilitates clear organization within the project’s modular data science architecture.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\GermanCreditDataClassifier/blob/master/src\module_training.py'>module_training.py</a></b></td>
					<td style='padding: 8px;'>- Handles the process of training a machine learning model on provided data while measuring and reporting the time taken for training<br>- By encapsulating model training and performance monitoring, it supports reproducibility and efficiency within the projects broader machine learning workflow, ensuring that model preparation is both streamlined and consistent with the codebases modular approach.</td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---
