
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import prompts.task_label

from prompts.task_label import TaskLabelPrompt

load_dotenv()


class ClassificationService:
    """
    Classification Service
    """

    @staticmethod
    def get_response_from_openai(user_message: str) -> str:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=task_label_prompt.compile_messages(user_message)
        )
        return response.choices[0].message.content

    @staticmethod
    def get_response_from_ollama(user_message: str) -> str:
        response = ollama_client.chat.completions.create(
            model="llama3.1:8b",
            messages=task_label_prompt.compile_messages(user_message)
        )
        return response.choices[0].message.content

    @staticmethod
    def evaluate_predictions(tasks: list[dict]) -> dict:
        """Evaluate model predictions using various metrics."""

        # Extract true labels and predictions
        y_true = [task['category'] for task in tasks]
        y_pred = [task['prediction'] for task in tasks]

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['work', 'home', 'errands', 'general'],
                    yticklabels=['work', 'home', 'errands', 'general'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(script_dir, "confusion_matrix.png"))

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        }