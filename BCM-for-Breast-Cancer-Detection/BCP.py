import numpy as np # This imports the NumPy library, which provides support for arrays and matrices of numerical data. NumPy is a fundamental package for scientific computing in Python.
import pandas as pd # This imports the Pandas library, which provides data structures for effectively working with and manipulating data in Python. Pandas is particularly useful for working with tabular data.
import seaborn as sns # This imports the Seaborn library, which provides a high-level interface for creating informative and attractive statistical graphics in Python. Seaborn is built on top of Matplotlib and is particularly useful for visualizing statistical relationships in data.
import matplotlib.pyplot as plt # This imports the Matplotlib library, which provides a comprehensive set of tools for creating static, animated, and interactive visualizations in Python. Matplotlib is the most widely used plotting library in Python.
from sklearn.datasets import load_breast_cancer # This imports the load_breast_cancer function from the Scikit-learn library, which provides a collection of tools for machine learning in Python. load_breast_cancer is a pre-defined dataset of breast cancer patients' information.
from sklearn.model_selection import train_test_split, cross_val_score # This imports the train_test_split and cross_val_score functions from Scikit-learn's model_selection module. train_test_split is a utility function for splitting data into training and testing sets, while cross_val_score is a function for performing cross-validation on a model.
from sklearn.preprocessing import StandardScaler # This imports the StandardScaler class from Scikit-learn's preprocessing module. StandardScaler is a method for standardizing data by removing the mean and scaling to unit variance.
from sklearn.linear_model import LogisticRegression # This imports the LogisticRegression class from Scikit-learn's linear_model module. LogisticRegression is a classification algorithm for predicting the probability of a binary outcome (e.g., 0 or 1).
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix # This imports several metrics for evaluating the performance of a classification model. accuracy_score computes the accuracy of a model's predictions, while roc_curve and auc are used to generate and evaluate the Receiver Operating Characteristic (ROC) curve. confusion_matrix is a way of evaluating the number of true positives, true negatives, false positives, and false negatives in a classification problem.

"""
This code defines a class called BinaryClassificationModel which performs binary classification on a pre-defined dataset of breast cancer patients' information.
The class initializes with the dataset, which is a Bunch object containing both the features and target. The dataset is split into training and testing sets using train_test_split.
The training and testing features are then scaled using StandardScaler and the LogisticRegression algorithm is used to fit the model.

The class contains methods for visualizing the class distribution, preprocessing the data, training the model, and evaluating the model.
The evaluate_model method generates and displays various evaluation metrics and plots, including the accuracy score, ROC curve, confusion matrix, and any predicted cancer cases.
The cross_val method performs cross-validation on the model.
"""

class BinaryClassificationModel:
    def __init__(self, data, max_iter=3000):
        """
        Initializes a BinaryClassificationModel object.

        Args:
            data (sklearn.datasets.base.Bunch): A bunch object containing both the features and target.
            max_iter (int, optional): Maximum number of iterations taken for the solver to converge. Defaults to 5000.
        """
        self.data = data
        self.df = pd.DataFrame(np.c_[data.data, data.target], columns=list(data.feature_names) + ['target'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(random_state=42, max_iter=max_iter, solver='lbfgs')

    def visualize_class_distribution(self):
        """
        Plots the count of instances for each target class in the dataset.
        """
        sns.set(style="whitegrid")  # set the style of the plot
        fig, ax = plt.subplots(figsize=(8, 6))  # set the size of the plot
    
        # create the countplot
        ax = sns.countplot(x='target', data=self.df, palette='Set2')
    
        # customize the plot
        ax.set_title('Distribution of Target Classes', fontsize=18, fontweight='bold')
        ax.set_xlabel('Target Class', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
    
        # add annotations to the bars
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.0f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', fontsize=12, color='black', xytext=(0, 10), 
                        textcoords='offset points')
    
        plt.tight_layout()  # adjust the spacing between subplots
        plt.show()  # display the plot

    def preprocess_data(self):
        """
        Scales the training and testing features using StandardScaler.
        """
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        """
        Trains the logistic regression model on the preprocessed training data.
        """
        self.classifier.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluates the logistic regression model on the preprocessed test data and
        generates and displays various evaluation metrics and plots, including the
        accuracy score, ROC curve, confusion matrix, and any predicted cancer cases.
        """
        y_pred = self.classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)

        y_pred_prob = self.classifier.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # create the ROC curve plot
        sns.set(style='ticks')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', color='#FFA500', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2)
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('Receiver Operating Characteristic', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc='lower right', fontsize=12)
        sns.despine()

        plt.tight_layout()
        plt.show()

        # create the confusion matrix plot
        confusion = confusion_matrix(self.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, cmap='Blues', cbar=False, fmt='g')
        ax.set_xlabel('Predicted Class', fontsize=14)
        ax.set_ylabel('True Class', fontsize=14)
        ax.set_title('Confusion Matrix', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.xaxis.set_ticklabels(['Benign', 'Malignant'], fontsize=12)
        ax.yaxis.set_ticklabels(['Benign', 'Malignant'], fontsize=12, rotation=0)
        plt.tight_layout()
        plt.show()

        # create the predicted cancer cases file
        df_pred = pd.DataFrame(np.c_[self.X_test, y_pred], columns=list(self.data.feature_names) + ['predicted_label'])
        cancer_cases = df_pred[df_pred['predicted_label'] == 1]
        if not cancer_cases.empty:
            cancer_cases.to_csv('Predicted-Cancer-Cases.txt', sep='\t', index=True)
            print("Predicted cancer cases written to Predicted-Cancer-Cases.txt")
        else:
            print("No cancer cases predicted.")

    def cross_validate_model(self):
        """
        Performs 5-fold cross-validation on the logistic regression model and
        generates and displays the cross-validation scores and mean score.
        """
        cv_scores = cross_val_score(self.classifier, self.data.data, self.data.target, cv=5)
        print("Cross-validation Scores:", cv_scores)
        print("Mean Cross-validation Score:", np.mean(cv_scores))

if __name__ == "__main__":
    data = load_breast_cancer()
    model = BinaryClassificationModel(data, max_iter=3000)
    model.visualize_class_distribution()
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.cross_validate_model()