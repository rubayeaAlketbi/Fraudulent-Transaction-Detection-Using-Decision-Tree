# Fraudulent Transaction Detection Using Decision Tree

This project focuses on implementing various tasks related to decision tree classification using the `scikit-learn` library in Python. The goal is to process and analyze a dataset, train decision tree models, and optimize their performance. We will build a model that can detect fraudulent transactions. The steps involve:

1. **Reading and filtering the data**
2. **Splitting the data into training and testing groups**
3. **Training a decision tree**

After the decision tree is trained, we will prune it by increasing the cost complexity parameter, which helps optimize the model’s complexity. Different depths of the tree will be tested and compared, and while deeper trees may capture more complex patterns, they don’t always produce more accurate models.

### Dataset Overview

The dataset contains a series of transactions and associated features. Each row corresponds to a transaction, and the columns include the following:

- **Transaction Amount**: Amount involved in the transaction.
- **Time Of Day**: The time of day when the transaction occurred.
- **Transaction Frequency**: How many times a transaction happened in a row.
- **Day Of Week**: The day of the week when the transaction took place.
- **Labels**: Binary labels where 1 indicates a fraudulent transaction, and 0 indicates a legitimate one.

### Features Used in the Decision Tree Model

- **Transaction Amount**: The amount involved in the transaction.
- **Time Of Day**: The time at which the transaction occurred.
- **Day Of Week**: The day on which the transaction occurred.
- **Transaction Frequency**: Frequency of transactions within a period.

These features are input into the decision tree model, which outputs whether the transaction is fraudulent or not.
