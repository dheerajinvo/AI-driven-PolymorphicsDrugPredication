#  Drug–Drug Interaction & Side Effect Predictor using Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

##  Project Overview

This project predicts **drug–drug interactions and possible side effects** using a **Graph Neural Network (GNN)**.

The system analyzes drugs taken by patients and identifies potential harmful interactions between drug pairs.

The application is built using:

* **Graph Neural Networks (GNN)**
* **PyTorch & PyTorch Geometric**
* **Streamlit Web Interface**

---

## 🚀 Features

* Drug–drug interaction prediction
* Side effect detection
* Severity identification
* Patient drug interaction analysis
* Interactive Streamlit web application
* Upload custom datasets

---

##  Model

The model uses a **Graph Neural Network (GNN)** where:

* **Nodes** represent drugs
* **Edges** represent interactions between drugs

The network learns relationships between drugs to predict possible interactions.

---

##  Project Structure

```
Drug-Interaction-GNN
│
├── app.py
├── drug_interactions.csv
├── patients_dataset.csv
├── requirements.txt
├── README.md
└── project_guide.pdf
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/your-username/drug-interaction-gnn.git
```

Move to the project folder:

```
cd drug-interaction-gnn
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📊 Example Drug Interaction Dataset

```
Drug1,Drug2,severity,description
Aspirin,Warfarin,Severe,High bleeding risk
Ibuprofen,Warfarin,Moderate,Increased bleeding risk
Paracetamol,Warfarin,Mild,Liver enzyme interaction
Metformin,Atorvastatin,Moderate,Muscle toxicity risk
```

---

## 👨‍⚕️ Example Patient Dataset

```
Name,Drugs
John,Aspirin;Warfarin
Alice,Ibuprofen;Atorvastatin
Bob,Metformin;Insulin
```

Note: Multiple drugs must be separated with **semicolon ( ; )**

---

## 📈 Output

The system displays:

* Drug pairs
* Interaction probability
* Severity level
* Possible side effects

Example:

```
Drug1: Ibuprofen
Drug2: Atorvastatin
Interaction Probability: 0.91
Severity: Moderate
Possible Side Effect: Muscle toxicity risk
```

---

## 🖥️ Application Interface

(Add screenshot here)

Example:

```
![App Screenshot](images/app_interface.png)
```

---

## 🛠️ Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## 📚 Future Improvements

* Larger drug interaction dataset
* Graph Attention Networks (GAT)
* Drug interaction visualization
* Explainable AI for drug predictions
* Integration with medical drug databases

---

## 📄 License

This project is for educational and research purposes.
