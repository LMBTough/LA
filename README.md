# 🧠 Local Attribution (LA)

This repository provides the **official implementation** of the paper:

> **Enhancing Model Interpretability with Local Attribution over Global Exploration**  
> In *Proceedings of the 32nd ACM International Conference on Multimedia (ACM MM 2024)*  
> [https://dl.acm.org/doi/abs/10.1145/3664647.3681385]  

We provide a unified framework for generating, evaluating, and visualizing **local attribution** across a wide range of interpretability methods on ImageNet models.

This library is designed to support systematic comparison and reproducibility of attribution methods, especially focusing on the local region-based enhancements proposed in our LA method.


---

## 🔧 Requirements

Create a virtual environment and install the dependencies:

```bash
conda env create -f environment.yml
conda activate LA
```

or

```bash
conda create -n LA python=3.11 -y
conda activate LA
pip install -r requirements.txt
```

---

## 📂 Directory Structure

```
LA/
├── generate_pt.py               # Generate feature files (.pt)
├── generate_attributions.py     # Compute attribution maps
├── eval.py                      # Evaluate attribution scores (Insertion, Deletion)
├── plot.py                      # Basic visualization
├── para_explore.sh              # Grid search script for LA parameters
├── saliency/                    # Contains attribution methods
│   ├── saliency_zoo.py
│   └── core/
│       ├── ig.py, la.py, agi.py, ...
├── data/
│   └── label_batch.pt           # Labels for evaluation
├── visualized_imgs/
│   ├── 0.png, 1.png, 2.png, ... # dataset original images
├── utils.py, resnet_mod.py, vgg16_mod.py
├── imagenet_class_index.json
├── plot_hm.ipynb                # Demo of drawing heatmap results
└── README.md                    
```

---

## 🚀 Usage

### 1. Generate Intermediate Features

```bash
python generate_pt.py
```

This will save `.pt` files containing image features and labels used for attribution.

---

### 2. Generate Attribution Maps

```bash
python generate_attributions.py --model resnet50 --attr_method la --spatial_range 20 --max_iter 20 --sampling_times 20
```

Replace `la` with any of the supported attribution methods:

```
["fast_ig", "deeplift", "guided_ig", "ig", "sg", "big", "sm", "mfaba", "eg", "agi", "attexplore", "la"]
```

Note: Only `la` uses the additional hyperparameters: `--spatial_range`, `--max_iter`, `--sampling_times`.

---

### 3. Evaluate Attribution Scores

```bash
python eval.py --model resnet50 --attr_method la --max_iter 20 --spatial_range 20 --samples_number 20 --prefix scores --csv_path results.csv --attr_prefix attributions
```

This calculates the **Insertion** and **Deletion** metrics and appends results to `results.csv`.

If `attr_method` is not `la`, the corresponding spatial parameters will be marked as `"-"` in the results file.

---


### 4. Visualize Attribution Maps

To visualize attribution results, we recommend using the provided Jupyter notebook:

```bash
jupyter notebook plot_hm.ipynb
```

> **Note**: To reproduce the full set of visualizations shown in the paper, you will need to **precompute the required attribution maps** using:
>
> ```bash
> bash para_explore.sh
> ```
>
> This script generates all attribution results (including baselines and our LA method under different settings) necessary for rendering the visualizations.
> It ensures consistency with the figures reported in the original paper.

If you only need to visualize a specific method (e.g., LA), we recommend simplifying `plot_hm.ipynb` accordingly.

---


### 5. Grid Search (Optional)

To perform parameter exploration for `la`, use the script:

```bash
bash para_explore.sh
```

---

## 📜 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

---

### 📚 Citation

If you find this repository helpful in your research, please consider citing our paper:

```bibtex
@inproceedings{zhu2024enhancing,
  title={Enhancing model interpretability with local attribution over global exploration},
  author={Zhu, Zhiyu and Jin, Zhibo and Zhang, Jiayu and Chen, Huaming},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={5347--5355},
  year={2024}
}
```

---
