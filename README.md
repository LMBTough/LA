
---

# LA

## 🔧 Requirements

Make sure you have installed all the necessary dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Generate Attribution Results

To generate attribution results using a specific attribution method (e.g., Layer-wise Attention):

```bash
python generate_attributions.py --attr_method la
```

Replace `la` with any other supported attribution method as needed.

### 2. Evaluate Attribution (Insertion/Deletion)

To evaluate the quality of the attributions using the insertion and deletion metrics:

```bash
python eval.py --attr_method la
```

This will compute performance metrics to assess how well the attribution highlights important regions.

### 3. Visualize Attribution

To visualize the attribution on the image:

```python
from plot import plot

plot(attribution, image)
```

This will generate an overlay showing the attribution map on top of the original image.

---
