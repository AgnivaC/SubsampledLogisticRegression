# A Provably Accurate Randomized Sampling Algorithm for Logistic Regression
 Code repository for the paper:

> <a href="https://agnivac.github.io/">Agniva Chowdhury</a> and Pradeep Ramuhalli. <em>A Provably Accurate Randomized Sampling Algorithm for Logistic Regression</em>. In Proceedings of the 38th AAAI Conference on Artificial Intelligence, 2024.

### Technical Appendix

Technical Appendix of the paper can be found in <tt>TechnicalAppendix.pdf</tt>.

### Datasets

<ol>
<li>Cardiovascular disease dataset (cardio): <tt>cardio_train.csv</tt> (sourced from <a href="https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset">here</a>)</li>
<li>Bank customer churn prediction dataset (churn): <tt>Bank Customer Churn Prediction.csv</tt> (sourced from <a href="https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction">here</a>)</li>
<li>Default of credit card clients dataset (default): <tt>default of credit card clients.csv</tt> (sourced from <a href="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients">here</a>)</li>
</ol>


### Codes

<ol>
<li>To compute row leverage scores of a matrix: <tt>leverage_scores.py</tt></li>
<li>To perform leverage score, l2s, or uniform sampling: <tt>row_sampling.py</tt></li>
</ol>

The code for l2s sampling has been sourced from <a href="https://github.com/Tim907/oblivious_sketching_varreglogreg/blob/main/sketching/l2s_sampling.py">here</a>.

### Notebooks

To reproduce the experiments in the paper, run the following *Jupyter Notebooks*:
<ol>
<li>For Cardiovascular disease dataset: <tt>cardio_train.ipynb</tt></li>
<li>For Bank customer churn prediction dataset: <tt>default_of_credit_card_clients.ipynb</tt></li>
<li>For Default of credit card clients dataset: <tt>Bank_Customer_Churn_Prediction.ipynb</tt></li>
</ol>


### Citation

> [@article{Chowdhury_Ramuhalli_2024, 
>   title={A Provably Accurate Randomized Sampling Algorithm for Logistic Regression},
>   author={Chowdhury, Agniva and Ramuhalli, Pradeep}, 
>   volume={38}, 
>   url={https://ojs.aaai.org/index.php/AAAI/article/view/29042}, 
>   DOI={10.1609/aaai.v38i10.29042}, 
>   number={10}, 
>   journal={Proceedings of the AAAI Conference on Artificial Intelligence},  
>   year={2024}, 
>   month={Mar.}, 
>   pages={11597-11605} 
> }]


</br>
Please contact <a href="https://agnivac.github.io/">Agniva Chowdhury</a> for questions or comments.