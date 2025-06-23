# CCIM — Cognitive Communication and Integration Model  
**Graph-Based Simulation of Cognitive Information Transmission**

---

## 🗂️ Project Overview

This project implements a full simulation of how structured information propagates between two cognitive entities, modeled as brain-like graphs. The system combines principles from:

- Shannon's Information Theory  
- Hebbian Learning mechanisms  
- Friston's Free Energy Principle  
- Graph theory applied to cognitive memory structures  

The simulation illustrates how information is selected, transmitted, implanted, and structurally integrated into the receiver's cognitive graph, including semantic, logical, and structural components.

---

## ⚙️ Requirements

Python 3.8+  
Required packages:

```bash
pip install networkx numpy matplotlib scipy scikit-learn python-louvain pyvis ipython nbformat nbconvert
```

---

## 🧪 Quick Start

### 1. Adjust Configuration

Edit the `config` dictionary at the top of the notebook or `.py` file:

```python
config = {
    "sender_size": 30,
    "receiver_size": 30,
    "transmitting": 5,
    "transmission_prob": 1,
    "chaos_level": 1
}
```

Where:

- `sender_size` – number of nodes in the sender brain  
- `receiver_size` – number of nodes in the receiver brain  
- `transmitting` – number of nodes in the message  
- `transmission_prob` – probability of successful transmission per node  
- `chaos_level` – 0 = structured message, 1 = random message  

---

### 2. Run the Full Simulation

Execute the notebook `brain.ipynb` or the exported `.py` file to:

✅ Generate cognitive graphs for sender and receiver  
✅ Create a structured or chaotic message subgraph  
✅ Simulate probabilistic transmission with semantic classification  
✅ Implant and integrate the message into the receiver's graph  
✅ Apply Hebbian learning to strengthen key connections  
✅ Visualize and analyze all graphs and metrics  

---

## 📁 Output Structure

All results are saved in:

```
/report/{sender_size}_{receiver_size}_{transmitting}_{transmission_prob}_{chaos_level}/
```

Key contents:

- `brain_state/` – visualizations of graphs before and after integration  
- `weights/` – CSV files with edge weights  
- `info_scores/` – information scores of transmitted nodes  
- `interactive/` – interactive PyVis visualizations  
- `compare_weight/` – weight histograms before and after integration  
- `dissociation_analysis/` – dissociation risk report  
- `config.json` – saved simulation parameters  

---

## 🔊 Interactive Graphs

Generate interactive browser-based visualizations:

```python
show_interactive_graph(receiver_brain_after)
```

Or highlight implanted nodes and clusters:

```python
show_interactive_receiver_brain_cluster_implanted(receiver_brain_after, config)
```

---

## 📊 Included Analyses

- Visualizations of sender, receiver, and transmitted graphs  
- Information score comparisons before and after integration  
- Predictive weight histograms (cosine similarity)  
- Cluster distributions and assignments  
- Embedding space projections (PCA)  
- Structural comparisons (Jaccard similarity, edit distance)  
- Dissociation risk evaluation  
- Free Energy measurement for semantic coherence  

---

## 🧬 Theoretical Foundations

This system integrates:

- **Shannon (1948)** – Probabilistic communication model  
- **Hebb (1949)** – Learning through connection strengthening  
- **Friston (2005)** – Predictive coding and Free Energy Principle  
- **Graph Theory** – Memory and cognition as dynamic networks  

---

## 🌐 Suggested Use Cases

- Cognitive modeling of memory integration  
- Simulation of neural information propagation  
- Educational demos for information theory and cognitive science  
- Testing robustness of graph structures under semantic implantation  

---

## 📅 License & Credits

Developed by Kacper Ragankiewicz 
For academic and research use only.
License: MIT

---
