import numpy as np
import pickle
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from glob import glob

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
    img = cv2.imread('gestures/1/100.jpg', 0)
    return img.shape

def get_num_of_classes():
    return len(glob('gestures/*'))

def preprocess_image(img):
    image_x, image_y = get_image_size()
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def load_test_data():
    print("Loading test data...")
    
    # Load test images and labels
    with open("test_images", "rb") as f:
        test_images = np.array(pickle.load(f))
    with open("test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)
    
    # Reshape images
    image_x, image_y = get_image_size()
    test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
    
    print(f"Test data loaded: {len(test_images)} images")
    return test_images, test_labels

def get_class_names():
    """Get class names from the database or gesture folder names"""
    try:
        import sqlite3
        conn = sqlite3.connect("gesture_db.db")
        cursor = conn.execute("SELECT g_id, g_name FROM gesture ORDER BY g_id")
        class_names = {}
        for row in cursor:
            class_names[row[0]] = row[1]
        conn.close()
        return [class_names.get(i, f"Class {i}") for i in range(get_num_of_classes())]
    except:
        # If database not available, use folder names
        class_names = []
        for i in range(get_num_of_classes()):
            class_names.append(f"Gesture {i}")
        return class_names

def calculate_metrics(model, test_images, test_labels):
    print("Calculating metrics...")
    
    # Get predictions
    predictions = model.predict(test_images)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, average='weighted')
    recall = recall_score(test_labels, y_pred, average='weighted')
    f1 = f1_score(test_labels, y_pred, average='weighted')
    
    # Create confusion matrix
    cm = confusion_matrix(test_labels, y_pred)
    
    # Get class report
    class_report = classification_report(test_labels, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'y_true': test_labels,
        'y_pred': y_pred
    }

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")

def plot_metrics_over_thresholds(model, test_images, test_labels):
    """Plot metrics over different confidence thresholds"""
    thresholds = np.arange(0.1, 1.0, 0.1)
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    
    predictions = model.predict(test_images)
    
    for threshold in thresholds:
        # Apply threshold
        y_pred = np.zeros_like(test_labels)
        for i, pred in enumerate(predictions):
            max_value = np.max(pred)
            max_idx = np.argmax(pred)
            if max_value >= threshold:
                y_pred[i] = max_idx
            else:
                # If below threshold, mark as "unknown" (could be a special class or handled differently)
                y_pred[i] = -1
        
        # Only evaluate on predictions above threshold
        mask = y_pred != -1
        if np.sum(mask) > 0:
            accuracies.append(accuracy_score(test_labels[mask], y_pred[mask]))
            precisions.append(precision_score(test_labels[mask], y_pred[mask], average='weighted'))
            recalls.append(recall_score(test_labels[mask], y_pred[mask], average='weighted'))
            f1s.append(f1_score(test_labels[mask], y_pred[mask], average='weighted'))
        else:
            accuracies.append(0)
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, 'o-', label='Accuracy')
    plt.plot(thresholds, precisions, 'o-', label='Precision')
    plt.plot(thresholds, recalls, 'o-', label='Recall')
    plt.plot(thresholds, f1s, 'o-', label='F1-Score')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Confidence Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('metrics_vs_threshold.png')
    print("Threshold analysis saved as 'metrics_vs_threshold.png'")

def evaluate_per_class(metrics, class_names):
    """Print per-class metrics summary"""
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    
    print("\nPer-class Performance:")
    print("-" * 60)
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)
    
    for class_idx in range(len(class_names)):
        # Get indices where the true class is the current class
        class_indices = y_true == class_idx
        if np.sum(class_indices) == 0:
            continue
            
        # Calculate true positives
        true_positives = np.sum((y_true == class_idx) & (y_pred == class_idx))
        
        # Calculate false positives and false negatives
        false_positives = np.sum((y_true != class_idx) & (y_pred == class_idx))
        false_negatives = np.sum((y_true == class_idx) & (y_pred != class_idx))
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(class_indices)
        
        print(f"{class_names[class_idx]:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
    
    print("-" * 60)

def plot_misclassified_examples(model, test_images, test_labels, class_names, max_examples=10):
    """Plot some examples of misclassified images"""
    predictions = model.predict(test_images)
    y_pred = np.argmax(predictions, axis=1)
    
    # Find misclassified examples
    misclassified_indices = np.where(y_pred != test_labels)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified examples found!")
        return
    
    # Select a subset of misclassified examples
    num_examples = min(max_examples, len(misclassified_indices))
    selected_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
    
    # Plot the examples
    rows = (num_examples + 4) // 5  # Ceiling division to determine number of rows
    cols = min(5, num_examples)     # At most 5 columns
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
    
    for i, idx in enumerate(selected_indices):
        if i >= len(axes):
            break
            
        img = test_images[idx].reshape(test_images[idx].shape[0], test_images[idx].shape[1])
        axes[i].imshow(img, cmap='gray')
        
        true_label = class_names[test_labels[idx]]
        pred_label = class_names[y_pred[idx]]
        confidence = np.max(predictions[idx]) * 100
        
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%")
        axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(selected_indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    print("Misclassified examples saved as 'misclassified_examples.png'")

def main():
    # Load model
    print("Loading model...")
    try:
        model = load_model('cnn_model_keras2.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test data
    test_images, test_labels = load_test_data()
    
    # Get class names
    class_names = get_class_names()
    
    # Calculate metrics
    metrics = calculate_metrics(model, test_images, test_labels)
    
    # Print overall metrics
    print("\n=== Overall Model Performance ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Print detailed classification report
    print("\n=== Classification Report ===")
    print(metrics['classification_report'])
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    
    # Evaluate per-class performance
    evaluate_per_class(metrics, class_names)
    
    # Plot metrics over thresholds
    plot_metrics_over_thresholds(model, test_images, test_labels)
    
    # Plot misclassified examples
    plot_misclassified_examples(model, test_images, test_labels, class_names)
    
    print("\nEvaluation complete! Results saved to disk.")

if __name__ == "__main__":
    main()