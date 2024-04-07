# COMP2611-Artificial Intelligence-Coursework#2 - Descision Trees

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.tree import export_text
import warnings
import os

# STUDENT NAME: Rubayea Heyab Salem AlKetbi
# STUDENT EMAIL: sc22rhsm@leeds.ac.uk
    
def print_tree_structure(model, header_list):
    tree_rules = export_text(model, feature_names=header_list[:-1])
    print(tree_rules)
    
# Task 1 [10 marks]: Load the data from the CSV file and give back the number of rows
def load_data(file_path, delimiter=','):
    num_rows, data, header_list=None, None, None
    if not os.path.isfile(file_path):
        warnings.warn(f"Task 1: Warning - CSV file '{file_path}' does not exist.")
        return None, None, None
    try:
        # Read the data from the CSV file
        data = pd.read_csv(file_path, delimiter=delimiter)
        # Get the number of rows and header list of the data
        num_rows = data.shape[0] 
        header_list = list(data.columns) 
    # Handle any exceptions that may occur during reading the CSV file
    except Exception as e:
        warnings.warn(f"Task 1: Warning - Error in reading the CSV file '{file_path}'. {e}")
        return None, None, None
    # Return the number of rows, data, and header list
    return num_rows, data, header_list

# Task 2[10 marks]: Give back the data by removing the rows with -99 values 
def filter_data(data):
    filtered_data=[None]*1
    # Store the indices to drop
    indices_to_drop = []
    # Iterate over DataFrame rows
    for index, row in data.iterrows():
        # Check if -99 is in the row values
        if -99 in row.values:
            # Add the index to the list of indices to drop
            indices_to_drop.append(index)     
    # Drop the rows
    filtered_data = data.drop(indices_to_drop)
    return filtered_data

# Task 3 [10 marks]: Data statistics, return the coefficient of variation for each feature, make sure to remove the rows with nan before doing this. 
def statistics_data(data):
    coefficient_of_variation=None
    data=filter_data(data)
    # removes the rows with nan values
    data = data.dropna()
    # get the mean and standard deviation of the data 
    mean = data.mean() 
    std = data.std()
    # Calculate the coefficient of variation for each feature
    coefficient_of_variation = std/mean 
    return coefficient_of_variation

# Task 4 [10 marks]: Split the dataset into training (70%) and testing sets (30%), 
# use train_test_split of scikit-learn to make sure that the sampling is stratified, 
# meaning that the ratio between 0 and 1 in the lable column stays the same in train and test groups.
# Also when using train_test_split function from scikit-learn make sure to use "random_state=1" as an argument. 
def split_data(data, test_size=0.3, random_state=1):
    x_train, x_test, y_train, y_test=None, None, None, None
    np.random.seed(1)
    # Split the data into labels and features. X : Features , Y : Labels
    # Select all columns except the last one as features
    X = data.iloc[:, :-1]
    # Select the last column as the label  
    y = data.iloc[:, -1]   
    # Split the data into training and testing sets, ensuring stratification
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    return x_train, x_test, y_train, y_test

# Task 5 [10 marks]: Train a decision tree model with cost complexity parameter of 0
def train_decision_tree(x_train, y_train,ccp_alpha=0):
    model=None
    # Initialize the DecisionTreeClassifier with the specified ccp_alpha
    model = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=1)
    # Fit (train) the model on the training data
    model.fit(x_train, y_train)
    return model

# Task 6 [10 marks]: Make predictions on the testing set 
def make_predictions(model, X_test):
    y_test_predicted=None
    y_test_predicted = model.predict(X_test)
    return y_test_predicted

# Task 7 [10 marks]: Evaluate the model performance by taking test dataset and giving back the accuracy and recall 
def evaluate_model(model, x, y):
    y_pred = make_predictions(model, x)
    # Calculate accuracy by comparing the predicted labels with the true labels
    accuracy = accuracy_score(y, y_pred)
    # Calculate recall by comparing the predicted labels with the true labels
    recall = recall_score(y, y_pred)
    return accuracy, recall

# Task 8 [10 marks]: Write a function that gives the optimal value for cost complexity parameter
# which leads to simpler model but almost same test accuracy as the unpruned model (+-1% of the unpruned accuracy)
def optimal_ccp_alpha(x_train, y_train, x_test, y_test):
    optimal_ccp_alpha=None

    # Train an unpruned decision tree to establish baseline accuracy
    tree_unpruned = DecisionTreeClassifier(random_state=0)
    tree_unpruned.fit(x_train, y_train)
    baseline_accuracy = accuracy_score(y_test, tree_unpruned.predict(x_test))
    
    # Initialize variables to track the best ccp_alpha
    best_ccp_alpha = 0
    last_accuracy = baseline_accuracy
    
    # Start with a small ccp_alpha and increment it to find the optimal value
    ccp_alpha_increment = 0.001  # Increment value for ccp_alpha
    ccp_alpha = ccp_alpha_increment  # Starting value
    
    # Loop to find the optimal ccp_alpha
    while True:
        tree = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        tree.fit(x_train, y_train)
        current_accuracy = accuracy_score(y_test, tree.predict(x_test))
        
        # Check if the accuracy drop is more than 1% from the baseline
        if (baseline_accuracy - current_accuracy) > 0.01:
            # If the accuracy drop is more than 1%, break the loop
            break  
        
        # Update best_ccp_alpha and last_accuracy
        best_ccp_alpha = ccp_alpha
        last_accuracy = current_accuracy
        
        # Increment ccp_alpha for the next iteration
        ccp_alpha += ccp_alpha_increment
    
    return best_ccp_alpha

# Task 9 [10 marks]: Write a function that gives the depth of a decision tree that it takes as input.
def tree_depths(model):
    depth=None
    # Get the depth of the unpruned tree
    depth = model.get_depth()
    return depth

 # Task 10 [10 marks]: Feature importance 
def important_feature(x_train, y_train,header_list):
    best_feature=None
    # Train decision tree model and increase Cost Complexity Parameter until the depth reaches 1
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(x_train, y_train)
   # Calculate the cost complexity pruning path
    path = tree.cost_complexity_pruning_path(x_train, y_train)
    cpp_alphas = path.ccp_alphas
    
    # Iterate over the ccp_alphas to find the best feature
    for ccp_alpha in cpp_alphas:
        # Train the decision tree with the current ccp_alpha
        tree = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        tree.fit(x_train, y_train)
        # Check if the tree length is 1
        if tree.get_depth() == 1:
            break
        
    # Get the feature importance
    best_feature = header_list[np.argmax(tree.feature_importances_)]
    return best_feature




'''Main Section'''
# Example usage (Template Main section):
if __name__ == "__main__":
    # Load data
    file_path = "DT.csv"
    num_rows, data, header_list = load_data(file_path)
    print(f"Data is read. Number of Rows: {num_rows}"); 
    print("-" * 50)

    # Filter data
    data_filtered = filter_data(data)
    num_rows_filtered=data_filtered.shape[0]
    print(f"Data is filtered. Number of Rows: {num_rows_filtered}"); 
    print("-" * 50)

    # Data Statistics
    coefficient_of_variation = statistics_data(data_filtered)
    print("Coefficient of Variation for each feature:")
    for header, coef_var in zip(header_list[:-1], coefficient_of_variation):
        print(f"{header}: {coef_var}")
    print("-" * 50)
    
    # Split data
    x_train, x_test, y_train, y_test = split_data(data_filtered)
    print(f"Train set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print("-" * 50)
    
    # Train initial Decision Tree
    model = train_decision_tree(x_train, y_train)
    print("Initial Decision Tree Structure:")
    print_tree_structure(model, header_list)
    print("-" * 50)
    
    # Evaluate initial model
    acc_test, recall_test = evaluate_model(model, x_test, y_test)
    print(f"Initial Decision Tree - Test Accuracy: {acc_test:.2%}, Recall: {recall_test:.2%}")
    print("-" * 50)
    
    # Train Pruned Decision Tree
    model_pruned = train_decision_tree(x_train, y_train, ccp_alpha=0.002)
    print("Pruned Decision Tree Structure:")
    print_tree_structure(model_pruned, header_list)
    print("-" * 50)
    
    # Evaluate pruned model
    acc_test_pruned, recall_test_pruned = evaluate_model(model_pruned, x_test, y_test)
    print(f"Pruned Decision Tree - Test Accuracy: {acc_test_pruned:.2%}, Recall: {recall_test_pruned:.2%}")
    print("-" * 50)
    
    # Find optimal ccp_alpha
    optimal_alpha = optimal_ccp_alpha(x_train, y_train, x_test, y_test)
    print(f"Optimal ccp_alpha for pruning: {optimal_alpha:.4f}")
    print("-" * 50)
    
    # Train Pruned and Optimized Decision Tree
    model_optimized = train_decision_tree(x_train, y_train, ccp_alpha=optimal_alpha)
    print("Optimized Decision Tree Structure:")
    print_tree_structure(model_optimized, header_list)
    print("-" * 50)
    
    # Get tree depths
    depth_initial = tree_depths(model)
    depth_pruned = tree_depths(model_pruned)
    depth_optimized = tree_depths(model_optimized)
    print(f"Initial Decision Tree Depth: {depth_initial}")
    print(f"Pruned Decision Tree Depth: {depth_pruned}")
    print(f"Optimized Decision Tree Depth: {depth_optimized}")
    print("-" * 50)
    
    # Feature importance
    important_feature_name = important_feature(x_train, y_train,header_list)
    print(f"Important Feature for Fraudulent Transaction Prediction: {important_feature_name}")
    print("-" * 50)
        
'''
References: 
 - Task 2-3 is referenced from pandas documentation at https://pandas.pydata.org/docs/
 - Line 65 , the coeficient of variation formula is referenced from https://en.wikipedia.org/wiki/Coefficient_of_variation
 - Line 76-78 is inspired from https://stackoverflow.com/questions/46519539/how-to-select-all-non-nan-columns-and-non-nan-last-column-using-pandas
 - Line 80-85 is inspired from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
 - Line 92 is inspired from https://scikit-learn.org/stable/modules/tree.html#decision-trees
 - Line 94 is inspired from https://www.geeksforgeeks.org/learning-model-building-scikit-learn-python-machine-learning-library/
 - Line 100 is inspired from https://www.askpython.com/python/examples/python-predict-function
 - Line 106-109 is inspired from https://www.linkedin.com/pulse/basics-decision-tree-python-omkar-sutar#:~:text=To%20calculate%20the%20accuracy%20score,from%20the%20scikit%2Dlearn%20library.&text=In%20this%20code%2C%20y_test%20is,by%20the%20decision%20tree%20model.
 - Line 154 is inspired from https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.get_depth
 
 '''