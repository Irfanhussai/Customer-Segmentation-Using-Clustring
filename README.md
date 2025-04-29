# 🧠 AI-Powered Customer Segmentation

This project performs intelligent customer segmentation using **deep learning**, **clustering**, and **visualization**. It uses an autoencoder to reduce feature dimensionality, **KMeans** for clustering, and **UMAP** for 2D projection — all wrapped into an interactive **Streamlit** app for real-time predictions.

---

## 🚀 Demo

> 🔗 Launch the app locally with:
```bash
streamlit run streamlit_app.py
```

---

## 📂 Project Structure

```
├── model_trainer.py         # Trains autoencoder, clusters customers, prepares UMAP projections
├── streamlit_app.py         # Streamlit frontend for prediction & visualization
├── Mall_Customers.csv       # Dataset
├── README.md                # Project description
```

---

## 📊 Dataset: [Mall_Customers.csv](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial)

- **Gender**
- **Age**
- **Annual Income (k$)**
- **Spending Score (1-100)**

---

## 🔍 What This Project Does

| Step | Description |
|------|-------------|
| 🔄 Preprocessing | Label encodes `Gender`, standardizes numeric features |
| 🧠 Autoencoder | Learns compressed latent representation of customer features |
| 🎯 Clustering | Uses `KMeans` on latent features to group customers |
| 📉 UMAP | Reduces dimensionality for interactive 2D visualization |
| 🧩 Streamlit App | User inputs customer profile → predicts & visualizes cluster |

---

## 📈 Clusters Explained

| Cluster | Description |
|--------:|:------------|
| 0 | **Cautious Elites** |
| 1 | **Impulsive Spenders** |
| 2 | **Budget Conscious** |
| 3 | **Young & Social** |
| 4 | **High-Value Loyalists** |

---

## 🌐 Technologies Used

- 🐍 Python
- 🤖 Keras (Autoencoder)
- 🎯 Scikit-learn (KMeans)
- 🌌 UMAP
- 🎨 Seaborn + Matplotlib
- ⚡ Streamlit

---

## 💪 How to Run

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Train the Model**
```bash
python model_trainer.py
```

3. **Launch App**
```bash
streamlit run streamlit_app.py
```

---

## 📌 Future Enhancements

- Save/load models (`.pkl`, `.h5`) instead of retraining every time
- Enable CSV upload for batch customer predictions
- Add more behavioral features (e.g., loyalty points, recency)

---

## 👨‍💼 Author

**Your Name**  
_Internship Project @ Roman Tech_  
📧 your.email@example.com  
🔗 [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/)

