
<h1 align="center">
Loans reliability prediction</h1>
<h3 align="center"> Computer, Electronic and Telecommunications Engineering </h3>
<h5 align="center"> Project Assignment - Introduction to artificial intelligence  - <a href="https://www.polimi.it/">University of Parma</a> (March 2023) </h5>

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project"> ➤ About The Project</a></li>
    <li><a href="#project-files-description"> ➤ Project Files Description</a></li>
    <li><a href="#dataset"> ➤ Dataset </a></li>
    <li><a href="#analysis"> ➤ Analysis</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ABOUT THE PROJECT -->
<h2 id="about-the-project"> :pencil: About The Project</h2>

<p align="justify"> 
 The primary source of income for banks comes from mortgage contracts. A mortgage is a contract in which a lender, typically a bank, transfers a certain amount of money to a borrower, and the borrower assumes the obligation of repayment. This type of loan is subject to risks, as the borrower may prove to be unreliable, failing to repay the money. The purpose of the presented project is to classify loan applicants as reliable or not using machine learning models trained to perform binary classification tasks, specifically supervised learning. These models are trained on the "Loan_Default.csv" dataset, which contains data on various borrowers collected over time.
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PROJECT FILES DESCRIPTION -->
<h2 id="project-files-description"> :floppy_disk: Project Files Description</h2>
<ul>
  <li><b>Constants.py</b>: File containing constants to configure the execution of the scripts.</li>
  <li><b>Loan_Default.csv</b>:the dataset from kaggle (<a href="https://www.kaggle.com/datasets/yasserh/loan-default-dataset">kaggle.com</a>)</li>
  <li><b>Loans.py</b>The main script</li>
  <li><b>MongoDB_Load.py</b>Integration script for MongoDB.</li>
  <li><b>Plot_utils.py</b>: Script for data visualization.</li>
  <li><b>Relazione.pdf</b>: The full analysis (discalimer: available only in italian).</li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2 id="about-the-project"> 💿: Dataset</h2>
The dataset (unprocessed) is quite extensive, containing 148,670 examples with a total of 33 features and 1 target variable, resulting in a matrix of size (148670 x 34). The dataset contains many missing values. The dataset contains both categorical variables (with values mostly expressed as strings) and numerical variables (some of which also have negative values). Given the dataset's characteristics, it requires pre-processing, particularly in terms of feature engineering and feature selection.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- Analysis -->
<h2 id="analysis"> :small_orange_diamond:Analysis</h2>
The analysis carried out is contained in the script Loans.py and is described in detail in the file Relazione.pdf. It contains the preprocesing on the dataset and the best model selection.
