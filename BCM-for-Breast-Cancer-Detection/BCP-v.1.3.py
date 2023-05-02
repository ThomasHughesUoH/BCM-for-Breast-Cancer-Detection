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
This code defines a class called breast_cancer_binary_classification_model which performs binary classification on a pre-defined dataset of breast cancer patients' information.
The class initializes with the dataset, which is a Bunch object containing both the features and target. The dataset is split into training and testing sets using train_test_split.
The training and testing features are then scaled using StandardScaler and the LogisticRegression algorithm is used to fit the model.

The class contains methods for visualizing the class distribution, preprocessing the data, training the model, and evaluating the model.
The evaluate_model method generates and displays various evaluation metrics and plots, including the accuracy score, ROC curve, confusion matrix, and any predicted cancer cases.
The cross_val method performs cross-validation on the model.
"""

class breast_cancer_binary_classification_model:
    def __init__(self, data, max_iter=3000):
        """
        Initializes a breast_cancer_binary_classification_model object.

        Args:
            data (sklearn.datasets.base.Bunch): A bunch object containing both the features and target.
            max_iter (int, optional): Maximum number of iterations taken for the solver to converge. Defaults to 3000.
        """
        self.data = data
        self.df = pd.DataFrame(np.c_[data.data, data.target], columns=list(data.feature_names) + ['target'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(random_state=42, max_iter=max_iter, solver='lbfgs')

    def visualise_class_distribution(self):
        """
        Plots the count of instances for each target class in the dataset.
        """
        sns.set(style="whitegrid", palette=['blue', 'grey'])  # set the style and color palette of the plot
        fig, ax = plt.subplots(figsize=(8, 6))  # set the size of the plot

        # create the countplot
        ax = sns.countplot(x='target', data=self.df)

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

    def evaluate_model(self, n=10):
        """
        Evaluates the performance of the logistic regression model on the testing data.
        
        Args:
            n (int, optional): Number of top correlated features to plot. Defaults to 10.
        """
        # Preprocess the data
        self.preprocess_data()
        
        # Train the model
        self.train_model()
        
        # Generate predictions on the testing data
        y_pred = self.classifier.predict(self.X_test)
        
        # Compute evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        # Print evaluation metrics
        print(f"Accuracy: {accuracy:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18, fontweight='bold')
        plt.legend(loc="lower right")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(linestyle='--', alpha=0.7)
        plt.show()

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.4)
        sns.set_style("ticks")
        sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='g',cmap=sns.color_palette(["#C9D9D3", "#2F4B7C"]))
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('True', fontsize=14)
        plt.title('Confusion Matrix', fontsize=18, fontweight='bold')
        plt.tick_params(axis='both', which='both', length=0)
        plt.show()

       # Plot bar chart of top n correlated features
        corr = self.df.corr()
        corr_target = abs(corr["target"])
        top_features = corr_target.sort_values(ascending=False)[1:n+1]
        colors = ['grey' if c < 0 else 'lightblue' for c in top_features.values]
        plt.figure(figsize=(12,8))
        plt.barh(y=top_features.index, width=top_features.values, color=colors)
        plt.xlabel('Correlation with Target', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.title(f'Top {n} Correlated Features', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
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
        
    def feature_selection_correlation(self):
        """
        Selects the most correlated features and compares them against each other.
        """
        # Correlation between features and target
        corr_with_target = self.df.corrwith(self.df['target']).abs().sort_values(ascending=False)

        # Select the top 10 features with the highest correlation with the target
        top_features = corr_with_target.iloc[1:11].index.values

        # Correlation matrix of the selected features
        corr_matrix = self.df[top_features].corr()

        # Set the style and create the heatmap
        sns.set(style="ticks", font_scale=1.2)
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=True, cmap="Blues", linewidths=0.5)

        # Set the title and axis labels
        plt.title("Correlation Matrix of the Most Correlated Features", fontsize=16, fontweight='bold')
        plt.xlabel('Features', fontsize=14, fontweight='bold')
        plt.ylabel('Features', fontsize=14, fontweight='bold')

        # Set tick label size
        plt.xticks(fontsize=9, rotation=45)
        plt.yticks(fontsize=9)

        # Display the plot
        plt.show()
        
if __name__ == "__main__":
    data = load_breast_cancer()
    model = breast_cancer_binary_classification_model(data, max_iter=3000)
    model.visualise_class_distribution()
    model.feature_selection_correlation()
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.cross_validate_model()
