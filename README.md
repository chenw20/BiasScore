# Modification Bias Score
This is the code for the paper [Post-hoc bias scoring is optimal for fair classification](https://openreview.net/forum?id=FM5xfcaR2Y)

To train the model, run main.py. For testing, run postprocess_dp.py when the fairness constraint is Demographic Parity (DP) or postprocess_eo.py when the fairness constraint is Equalized Odds (EO). These scripts take hyperparameters set in the config.yaml file.

Datasets: [Adult Census](https://www.kaggle.com/datasets/uciml/adult-census-income), [COMPAS](https://www.kaggle.com/datasets/danofer/compass) and [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)


## Citing the paper (bib)
```
@inproceedings{chen2024posthoc,
  title = {Post-hoc bias scoring is optimal for fair classification},
  author = {Chen, Wenlong and Klochkov, Yegor and Liu, Yang},
  booktitle = {International Conference on Learning Representations},
  year = {2024}
}
