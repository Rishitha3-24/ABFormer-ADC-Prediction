---
title: ABFormer
emoji: 🧬
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
# **ABFormer: A Transformer-Based Multi-Modal Framework for ADC Activity Prediction**

ABFormer is a multi-modal deep learning architecture designed for activity prediction and in-silico triage of Antibody–Drug Conjugates (ADCs). Unlike prior models—most notably ADCNet—that treat antibodies, antigens, linkers, and payloads as independent features, ABFormer introduces a contextualized antibody–antigen binding representation derived from a pretrained interaction encoder. This interface-aware design improves generalization to unseen antibody–antigen pairs and prevents the over-optimistic behaviour observed in baseline models.

---

## **Overview**

Antibody–Drug Conjugates combine a monoclonal antibody, a cleavable linker, and a cytotoxic payload into a single targeted therapeutic. Their biological activity depends on the integrated behaviour of these components, with antibody–antigen binding constituting the primary determinant of cellular uptake. Traditional computational models overlook this interface and perform naïve feature concatenation, limiting their capacity to distinguish active from inactive ADCs.

ABFormer addresses this limitation via interaction-centric transfer learning: antibody heavy chains and antigens are passed jointly through a pretrained bi-cross-attention encoder (AntiBinder), producing biologically meaningful embeddings that capture binding interface context.

---

## Dataset

ABFormer was trained and evaluated on ADCdb, a curated Antibody–Drug Conjugate dataset containing:

- Antibody heavy chain sequences
- Antibody light chain sequences
- Antigen sequences
- Linker structures
- Payload structures
- Drug-to-Antibody Ratio (DAR)
- Binary activity labels

The dataset was split using both random split and leave-pair-out evaluation protocols.


## Repository Structure

ABFormer/
├── app.py
├── train.py
├── inference.py
├── model.py
├── utils.py
├── AB_Data.py
├── data/
├── Embeddings/
├── ckpts/
├── Ablation/
└── architecture.png

## **Key Contributions**

* **Contextualized Antibody–Antigen Interface Encoding**
  ABFormer uses a frozen AntiBinder encoder to extract 2592-dimensional interaction-aware embeddings integrating ESM-2 sequence features and IgFold structural signals.

* **Small Molecules Processing**
  Linker and payload isoSMILES strings are embedded using a PyTorch re-implementation of FG-BERT (256-dimensional each), jointly fine-tuned during training. MACCS fingerprints provide complementary handcrafted features.

* **Structured Protein Representations**
  Antibody light chain and antigen sequences are embedded via ESM-2 (1280-dimensional each). AAC descriptors are used for Antibody heavy chain, light chain and antigen. MACCS fingerprints provide complementary handcrafted features.

* **Multi-Modal Fusion Architecture**
  All features—including DAR—are concatenated into a 6059-dimensional vector and passed through an MLP prediction head (6059 → 256 → 256 → 1) with GeLU activations.

* **Selective Fine-Tuning Strategy**
  AntiBinder, ESM-2, and IgFold remain frozen; FG-BERT and the MLP are trainable. This avoids overfitting in the chemically sparse ADCdb dataset.

---

## **Architecture**

![Architecture Image](architecture.png)

**Feature Encoders**

* AntiBinder (frozen): antibody–antigen interface → 2592-dim
* ESM-2 (frozen): light chain → 1280-dim, antigen → 1280-dim
* FG-BERT (fine-tuned): linker → 256-dim, payload → 256-dim
* MACCS (167-bit): linker, payload
* AAC descriptors: heavy chain, light chain, antigen
* DAR (scalar)

**Fusion**

* Concatenated 6059-dimensional vector
* Fully-connected MLP with GeLU activations
* Sigmoid output for binary activity classification

---

## Ablation Studies

To assess feature importance, several modalities were removed individually:

- Remove AAC
- Remove MACCS
- Remove DAR
- Remove Antigen
- Remove Chemical Embeddings
- Zero-shot Variant

Performance degradation observed during ablation experiments confirms the contribution of each modality to overall prediction performance.

## **Installation**

```bash
git clone https://github.com/drugparadigm/ABFormer.git
cd ABFormer
git lfs pull

conda env create -f ABFormer_env.yml
conda activate ABFormer
```

---

## **Usage**

### **Random Split Training**

```bash
python train.py
```

### **Leave-Pair-Out Split Training**

```bash
python train.py --unique_split
```

### **Inference on New Samples**

```bash
python inference.py --seed="$seed" --json_path="data.json"
```

## Deployment

ABFormer can be deployed as:

- Streamlit Application
- Docker Container
- Hugging Face Space

---
