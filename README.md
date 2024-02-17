# Modification Bias Score
This is the code for the paper [Post-hoc bias scoring is optimal for fair classification](https://openreview.net/forum?id=FM5xfcaR2Y)

To train the model, run main.py. For testing, run postprocess_dp.py when the fairness constraint is Demographic Parity (DP) or postprocess_eo.py when the fairness constraint is Equalized Odds (EO). These scripts take hyperparameters set in the config.yaml file.

Datasets: [Adult Census](https://www.kaggle.com/datasets/uciml/adult-census-income), [COMPAS](https://www.kaggle.com/datasets/danofer/compass) and [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

Dependencies:
- numpy - 1.24.2
- torch - 2.0.0+cu117
- torchvision - 0.15.1
- scikit-learn - 1.3.0
- scipy - 1.11.1
- pandas - 2.0.1
- Pillow - 9.5.0
- PyYAML - 6.0
- tqdm - 4.65.0
- typing_extensions - 4.5.0

## Citing the paper (bib)
```
@inproceedings{chen2024posthoc,
  title = {Post-hoc bias scoring is optimal for fair classification},
  author = {Chen, Wenlong and Klochkov, Yegor and Liu, Yang},
  booktitle = {International Conference on Learning Representations},
  year = {2024}
}
