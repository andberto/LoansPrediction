
<h1 align="center">European road network analysis</h1>
<h3 align="center"> Computer science and engineering </h3>
<h5 align="center"> Project Assignment - Complessità nei sistemi e nelle reti  - <a href="https://www.polimi.it/">Politecnico di Milano</a> (March 2023) </h5>

<p align="center"> 
  <img src="Images/flag.png" alt="Network Image" height="133" width="223">
</p>

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project"> ➤ About The Project</a></li>
    <li><a href="#project-files-description"> ➤ Project Files Description</a></li>
    <li><a href="#project-files-description"> ➤ Dataset </a></li>
    <li><a href="#Analysis summary"> ➤ Graph analysis</a></li>
    <li><a href="#references"> ➤ References</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ABOUT THE PROJECT -->
<h2 id="about-the-project"> :pencil: About The Project</h2>

<p align="justify"> 
 The primary source of income for banks comes from mortgage contracts. A mortgage is a contract in which a lender, typically a bank, transfers a certain amount of money to a borrower, and the borrower assumes the obligation of repayment. This type of loan is subject to risks, as the borrower may prove to be unreliable, failing to repay the money. The purpose of the presented project is to classify loan applicants as reliable or not using machine learning models trained to perform binary classification tasks, specifically supervised learning. These models are trained on the "Loan_Default.csv" dataset, which contains data on various borrowers collected over time.
</p>

The graph represents the European road network : <br>
<ul>
<li>A node represents a city. </li>
<li>An edge represents a road directly connecting two cities.</li>
</ul>
There are 1174 nodes (cities), connected by 1417 edges (direct roads), The network is undirected and unweighted.
<br>
The raw dataset can be found <a href="http://konect.cc/networks/subelj_euroroad/">here</a>, check also the profile of the author <a href="https://github.com/lovre">here</a>.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PROJECT FILES DESCRIPTION -->
<h2 id="project-files-description"> :floppy_disk: Project Files Description</h2>
<ul>
<li><b>GephiParser.py:</b> a simple script to parse some data in Gephy format.</li>
<li><b>CoordsRetriever.py:</b> it automatically retrieves cities coords using OpenStreetMaps API.</li>
<li><b>Euroroad.py:</b> the main script.</li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<h2 id="about-the-project"> 💿: Dataset</h2>
The dataset (unprocessed) is quite extensive, containing 148,670 examples with a total of 33 features and 1 target variable, resulting in a matrix of size (148670 x 34). The dataset contains many missing values. The dataset contains both categorical variables (with values mostly expressed as strings) and numerical variables (some of which also have negative values). Given the dataset's characteristics, it requires pre-processing, particularly in terms of feature engineering and feature selection.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- Analysis -->
<h2 id="graph-analysis"> :small_orange_diamond: Graph analysis</h2>

