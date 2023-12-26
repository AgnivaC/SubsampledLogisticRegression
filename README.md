# A Provably Accurate Randomized Sampling Algorithm for Logistic Regression
 Code repository for the paper:

> <a href="https://agnivac.github.io/">Agniva Chowdhury</a> and Pradeep Ramuhalli. <em>A Provably Accurate Randomized Sampling Algorithm for Logistic Regression</em>. In Proceedings of the 38th AAAI Conference on Artificial Intelligence, 2024.

### Datasets

<ol>
<li>Cardiovascular disease dataset (cardio): <tt>cardio_train.csv</tt></li>
<li>Bank customer churn prediction dataset (churn): <tt>Bank Customer Churn Prediction.csv</tt></li>
<li>Default of credit card clients dataset (default): <tt>default of credit card clients.csv</tt></li>
</ol>


### Codes

<ol>
<li>To compute row leverage scores of a matrix: <tt>leverage_scores.py</tt></li>
<li>To perform leverage score, l2s, or uniform sampling: <tt>row_sampling.py</tt></li>
</ol>

The code for l2s sampling has been sourced from <a href="https://github.com/Tim907/oblivious_sketching_varreglogreg/blob/main/sketching/l2s_sampling.py">here</a>.

### Notebooks