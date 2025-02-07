# Optimizing-MCTS-Game-Strategy-Predictions-with-Ensemble-Learning-

## 📌 Problem Statement
Monte Carlo Tree Search (MCTS) is a widely used search algorithm for developing intelligent board game agents. Over the past two decades, numerous MCTS variants have been proposed, but determining which variant is best suited for specific game types remains a challenge.

This competition aimed to develop a model that predicts the performance of one MCTS variant against another in a given game based on the features of the game. The evaluation metric used was **Root Mean Square Error (RMSE)** between the predicted and actual performance scores of the first agent against the second agent.

---

## 📊 Dataset
The dataset contains match outcomes between different MCTS agents across **1,000+ distinct two-player, sequential, zero-sum board games with perfect information**. The train dataset consists of features describing the game properties, agent types, and their respective performances. The final test dataset was evaluated using a hidden test set.

### 🔹 Key Features
- **GameRulesetName**: Unique combination of a game and its ruleset.
- **agent[1/2]**: String description of the MCTS agent configurations.
- **num_wins/draws/losses_agent1**: Number of wins/draws/losses for the first agent.
- **utility_agent1** *(Target Variable)*: The final score between -1 (all losses) and 1 (all wins).

📂 **Dataset Size**: ~1.35 GB

---

## ⚙️ Approaches & Methodology
I explored **four different approaches** to optimize the prediction model, experimenting with various feature engineering techniques, ensembling methods, and hyperparameter tuning.

### 🔥 Best Performing Approach: **Approach 2**
The best-performing model was developed in **Approach 2**, achieving an RMSE of **0.43008** on the private leaderboard, securing **Rank 755 / 1600** in the competition.

### 🚀 **Technical Stack Used in Approach 2**
- **Feature Engineering**:
  - Recursive Feature Elimination (RFE)
  - One-hot encoding of categorical variables
  - Target encoding of agent features
  - Feature scaling and normalization
- **Modeling Techniques**:
  - **Stacking Ensemble** of **LightGBM, CatBoost, XGBoost, and Neural Network**
  - Meta-model optimization using **Gradient Boosting**
  - **Bayesian Optimization** for hyperparameter tuning
  - **GroupKFold Cross-Validation** to handle game-specific variations
- **Performance Improvements**:
  - Reduced overfitting using **Regularization & Dropout**
  - **Variance Minimization Techniques** for better generalization
  - Optimized inference speed with **Numba & TQDM** for progress tracking

### 📌 Other Approaches (Brief Overview)
- **Approach 1**: Baseline model using LightGBM with minimal feature engineering.
- **Approach 3**: Added game-rule-based feature selection and experimented with deep learning models.
- **Approach 4**: Introduced model stacking with additional meta-features but faced overfitting issues.

---

## 📈 Results & Leaderboard Performance
- **Rank Achieved**: 🏆 **755 / 1600** participants
- **Best RMSE Score**: **0.43008** (Approach 2)

---

## 🔮 Future Work
- Implement **Transformer-based models** to capture deeper feature interactions.
- Fine-tune feature selection techniques for better generalization.
- Explore **LLM-assisted predictions** using pre-trained foundation models.

---

## 📌 How to Use
To reproduce the results, execute **Approach 2 notebook**:
```sh
# Install necessary libraries
pip install -r requirements.txt

# Run the best-performing model notebook
jupyter notebook um-game-playing-strength-of-mcts-approach-2.ipynb
```

---

## 📜 Acknowledgments
This work was developed as part of the Kaggle Competition **"UM Game Playing Strength of MCTS Variants"**. Special thanks to the Kaggle community for discussions and valuable insights.
Dataset:- https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/data
---

🎯 **If you find this work useful, don't forget to ⭐ the repository!** 🚀
