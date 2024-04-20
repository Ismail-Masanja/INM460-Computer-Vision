import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


__all__ = ['ModelEvaluator']


class ModelEvaluator:
    """ Evaluates the performance metrics of a model.

        REF: Originally from Ismail Masanja, N3063/INM702: Mathematics and Programming for AI.
            Modifications made to fit this project.
    """
    def __init__(self, model, dataloader, labels_map, device='cpu'):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.labels_map = labels_map  # Mapping of numeric labels to string labels
        self.device = device
    
    def evaluate(self):
        y_true = []
        y_pred = []
        
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(predicted.detach().cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Tabulate metrics
        print("Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Generate and print the classification report
        class_report = classification_report(y_true, y_pred, target_names=list(self.labels_map.values()), zero_division=0)
        print("\nClassification Report:\n", class_report)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_true, y_pred)
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=list(self.labels_map.keys()))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels_map.values(), yticklabels=self.labels_map.values())
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'{self.model.__class__.__name__} Confusion Matrix')
        plt.show()