# MSG: A Joint Masked Spectral Group Framework for Hyperspectral Representation Learning

## Description
Official implementation of MSG, a dual-branch self-supervised learning framework for hyperspectral image representation.
MSG introduces Spectral Group Embedding and Masking (SGEM) and jointly integrates masked reconstruction and contrastive learning for more transferable spectral–spatial representations.

---

## Highlights

- 🔹 **SGEM module**: partitions contiguous bands into groups, producing spectral–spatial tokens that keep inter-band correlation and spatial coherence.  
- 🔹 **Dual-branch SSL**: masked reconstruction (fidelity) + contrastive learning (discriminability) in a shared-encoder design.  
- 🔹 **Plug-and-play**: works with standard Transformer backbones.  
- 🔹 **Stronger transfer**: consistent gains under linear eval & fine-tuning across multiple HSI benchmarks.
