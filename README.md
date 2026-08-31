\# FIFA 21 Player Position Classification



Machine learning analysis comparing raw player attributes with PCA, Kernel PCA, and Locally Linear Embedding (LLE) for multiclass football position classification.



Using \*\*K-Nearest Neighbors (KNN)\*\* as a consistent classifier, the project evaluates whether dimensionality reduction can simplify FIFA 21 player attributes while preserving predictive performance.



\*\*Best held-out weighted F1: 0.8295\*\*



!\[Model Performance Comparison](figures/model\_performance\_comparison.png)



\---



\## Project Overview



Player positions are associated with different combinations of technical, physical, attacking, defensive, and movement attributes.



This project investigates two questions:



1\. Can FIFA 21 player attributes distinguish broad positional roles?

2\. Can dimensionality reduction compress the feature space without sacrificing classification performance?



Four feature representations are compared:



\- Raw standardized features

\- Principal Component Analysis (PCA)

\- Kernel PCA with an RBF kernel

\- Locally Linear Embedding (LLE)



KNN is used for every representation so that the experiment primarily measures how the representation itself affects classification.



\---



\## Dataset



The project uses the \*\*FIFA 21 Complete Player Dataset\*\* (`players\_21.csv`).



After preprocessing:



\- \*\*18,944 players\*\*

\- \*\*52 numeric features\*\*

\- \*\*6 position classes\*\*



\### Position Mapping



Detailed FIFA positions are grouped into six broader roles:



| Class | Included Positions |

|---|---|

| GK | GK |

| CB | CB |

| FB | LB, RB, LWB, RWB |

| CM | CDM, CM, CAM |

| Winger | LM, RM, LW, RW |

| ST | ST, CF |



For players with multiple listed positions, the dominant positional group is selected, with team position used as a tie-breaker when applicable.



\### Class Distribution



| Position | Players |

|---|---:|

| CM | 4,821 |

| Winger | 3,291 |

| CB | 3,152 |

| FB | 3,013 |

| ST | 2,583 |

| GK | 2,084 |



Because the classes are not perfectly balanced, \*\*weighted F1-score\*\* is used as the primary model-selection metric.



\---



\## Methodology



The dataset is split using stratified sampling:



\- \*\*Development set:\*\* 16,102 players

\- \*\*Held-out test set:\*\* 2,842 players



The held-out test set is not used for hyperparameter selection.



For each representation:



1\. Preprocessing is fit using development data.

2\. Hyperparameters are selected with stratified cross-validation.

3\. The selected configuration is refit on the full development set.

4\. Performance is evaluated on the held-out test set.



Scaling and learned transformations are fit only on the appropriate training data to prevent validation/test information from entering model fitting.



\### Cross-Validation



Raw features and PCA use \*\*5-fold stratified cross-validation\*\*.



Kernel PCA and LLE use \*\*3-fold stratified cross-validation\*\* because their transformations are substantially more computationally expensive.



For Kernel PCA and LLE, transformed representations are reused when evaluating different KNN neighborhood sizes to avoid repeatedly fitting the same expensive transformation.



\---



\## Models



\### Raw Features



Baseline pipeline:



```text

52 Features

&#x20;   ↓

StandardScaler

&#x20;   ↓

KNN

```



Selected configuration:



```text

KNN neighbors = 21

```



\### PCA



```text

52 Features

&#x20;   ↓

StandardScaler

&#x20;   ↓

PCA

&#x20;   ↓

KNN

```



Selected configuration:



```text

PCA components = 50

KNN neighbors  = 21

```



The selected 50 components preserve approximately \*\*99.996% of total variance\*\*.



\### Kernel PCA



```text

52 Features

&#x20;   ↓

StandardScaler

&#x20;   ↓

RBF Kernel PCA

&#x20;   ↓

KNN

```



Selected configuration:



```text

Components    = 50

Gamma         = 0.001

KNN neighbors = 30

```



\### Locally Linear Embedding



```text

52 Features

&#x20;   ↓

StandardScaler

&#x20;   ↓

LLE

&#x20;   ↓

KNN

```



Selected configuration:



```text

Components    = 50

LLE neighbors = 120

KNN neighbors = 30

```



\---



\## Final Results



| Representation | Selected Parameters | CV Weighted F1 | Test Accuracy | Test Weighted F1 |

|---|---|---:|---:|---:|

| Raw Features | k = 21 | \*\*0.8322\*\* | 0.8286 | 0.8287 |

| PCA | 50 components, k = 21 | \*\*0.8322\*\* | 0.8286 | 0.8287 |

| Kernel PCA | 50 components, γ = 0.001, k = 30 | 0.8304 | \*\*0.8293\*\* | \*\*0.8295\*\* |

| LLE | 50 components, 120 neighbors, k = 30 | 0.8280 | 0.8262 | 0.8269 |



Kernel PCA produced the highest numerical test weighted F1 at \*\*0.8295\*\*, but the difference from the raw-feature baseline was only \*\*0.0008\*\*.



The four representations therefore produced essentially equivalent held-out classification performance.



> The primary finding is not that dimensionality reduction improves classification, but that preserving baseline performance requires retaining nearly the entire original feature space.



\---



\## Dimensionality Reduction Analysis



PCA provides a useful view of how information is distributed throughout the dataset.



!\[PCA Explained Variance](figures/pca\_explained\_variance.png)



Although substantially fewer components capture most of the dataset's variance, the strongest classification performance requires nearly all 52 dimensions.



The selected PCA configuration uses:



```text

50 of 52 components

99.996% cumulative explained variance

```



This suggests that some relatively low-variance directions still contain information useful for distinguishing positions.



It also demonstrates an important distinction:



> \*\*Explained variance is not equivalent to predictive information.\*\*



PCA maximizes retained variance without considering the target class, so dimensions explaining relatively little overall variance may still help separate player positions.



\---



\## Class-Level Performance



!\[Per-Class F1 Comparison](figures/per\_class\_f1\_comparison.png)



Performance differs across positional groups because some roles have more distinctive attribute profiles than others.



Outfield positions share many characteristics — particularly pace, passing, dribbling, movement, attacking, and defensive attributes — creating overlap between neighboring positional roles.



The class-level analysis helps show that the overall \~0.83 weighted F1 is not distributed equally across every position.



\---



\## Error Analysis



Confusion matrices for each final model are available in the `figures/` directory:



\- `confusion\_matrix\_raw\_data.png`

\- `confusion\_matrix\_pca.png`

\- `confusion\_matrix\_kernel\_pca.png`

\- `confusion\_matrix\_lle.png`



These show which positional groups are most frequently confused and how similar the error patterns are across feature representations.



\---



\## Key Findings



\### 1. Raw features already provide a strong representation



KNN using the standardized 52-feature dataset achieves:



```text

Test Accuracy:    0.8286

Test Weighted F1: 0.8287

```



\### 2. PCA preserves performance but provides little compression



PCA exactly matches the raw-feature test performance, but only when retaining \*\*50 of 52 dimensions\*\*.



\### 3. Kernel PCA does not provide a meaningful predictive advantage



Kernel PCA achieves the highest numerical weighted F1:



```text

0.8295

```



compared with:



```text

0.8287

```



for the raw baseline.



The difference is too small to support a meaningful performance advantage.



\### 4. LLE also requires a high-dimensional representation



The best LLE configuration retains 50 dimensions and uses a relatively large neighborhood of 120 samples.



This suggests that a strongly compressed local manifold representation is not particularly advantageous for this task.



\### 5. Useful positional information is distributed across the feature space



All three dimensionality-reduction methods require nearly the complete original dimensionality to approach baseline performance.



This is the central result of the experiment.



\---



\## Repository Structure



```text

fifa-position-classification/

│

├── notebooks/

│   └── fifa\_position\_classification.ipynb

│

├── figures/

│   ├── model\_performance\_comparison.png

│   ├── pca\_explained\_variance.png

│   ├── per\_class\_f1\_comparison.png

│   ├── confusion\_matrix\_raw\_data.png

│   ├── confusion\_matrix\_pca.png

│   ├── confusion\_matrix\_kernel\_pca.png

│   └── confusion\_matrix\_lle.png

│

├── data/

│   └── README.md

│

├── README.md

├── requirements.txt

└── .gitignore

```



\---



\## Technologies



\- Python

\- pandas

\- NumPy

\- scikit-learn

\- Matplotlib

\- Jupyter Notebook / Google Colab

\- Git / GitHub



\### Machine Learning



\- K-Nearest Neighbors

\- StandardScaler

\- Principal Component Analysis

\- Kernel PCA

\- Locally Linear Embedding

\- GridSearchCV

\- StratifiedKFold



\---



\## Running the Project



\### 1. Clone the repository



```bash

git clone https://github.com/gabebekele/fifa-position-classification.git

cd fifa-position-classification

```



\### 2. Install dependencies



```bash

pip install -r requirements.txt

```



\### 3. Add the dataset



The dataset is not included in the repository.



Download the FIFA 21 player dataset and add `players\_21.csv` to the location expected by the notebook.



\### 4. Run the notebook



```bash

jupyter notebook notebooks/fifa\_position\_classification.ipynb

```



The notebook can also be run with Google Colab.



\---



\## Limitations



\- The six target classes simplify the more nuanced positional roles used in football.

\- FIFA ratings are video-game attributes rather than direct player-tracking or physical-performance measurements.

\- KNN performance depends heavily on feature scaling and the structure of the feature space.

\- PCA, Kernel PCA, and LLE provide little practical compression at their strongest configurations.

\- Raw/PCA and Kernel PCA/LLE use different cross-validation fold counts because of computational cost, so small differences in their CV scores should not be interpreted as direct model rankings.



\---



\## Future Work



Potential extensions include:



\- Compare KNN with SVM, Random Forest, and gradient-boosted classifiers

\- Compare dimensionality reduction with supervised feature selection

\- Evaluate alternative KNN distance metrics and weighting strategies

\- Use more granular positional classes

\- Evaluate models on another FIFA edition

\- Analyze which attributes contribute most strongly to positional separation

\- Compare predictive performance with computational cost across representations



\---



\## Conclusion



FIFA 21 player attributes can classify six broad positional roles with approximately \*\*83% held-out accuracy and weighted F1-score\*\*.



PCA, Kernel PCA, and LLE all preserve approximately the same classification performance as standardized raw features, but each requires \*\*50 of the original 52 dimensions\*\* at its strongest configuration.



The experiment therefore shows that aggressive dimensionality reduction does not benefit this task. Instead, positional information appears to be distributed across much of the original FIFA feature space, including dimensions that explain relatively little overall variance.

