import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import tensorflow as tf
import os

def train_with_stratified_kfold(X, y, label_encoder, build_model_fn, save_path=r"C:\Users\MSI\Desktop\Mitacs Project\Human Activity Recognition\UCI HAR Dataset\Models\Best_Model({fold}).h5", n_splits=5):
    """
    Train model using Stratified K-Fold cross-validation with CNN-LSTM architecture.

    Parameters:
    - X (np.ndarray): Preprocessed input data of shape (samples, timesteps, features)
    - y (np.ndarray): Encoded labels
    - label_encoder (LabelEncoder): LabelEncoder used for decoding classes
    - build_model_fn (function): Function that returns a compiled Keras model
    - save_path (str): Path to save the best performing model
    - n_splits (int): Number of folds for StratifiedKFold

    Returns:
    - None (saves model and prints/plots results)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores = []
    best_f1 = 0.0
    fold = 1
    ntimesteps = X.shape[1]
    nfeatures = X.shape[2]

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))

    for train_idx, val_idx in skf.split(X, y):
        print(f"\n📚 Training Fold {fold}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model_fn(input_shape=(ntimesteps, nfeatures), num_classes=len(np.unique(y)))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            class_weight=class_weight_dict,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Predictions
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)

        print(f"\n🧾 Fold {fold} Classification Report:")
        print(classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_.astype(str)))

        # Compute F1 score
        report = classification_report(y_val, y_pred_classes, output_dict=True)
        macro_f1 = report['macro avg']['f1-score']
        f1_scores.append(macro_f1)

        # Save best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            model.save(save_path)
            print(f"💾 Best model updated and saved to: {save_path}")

        # Plot accuracy/loss
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.plot(history.history.get('accuracy', []), label='Training Accuracy')
        plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
        plt.title(f'Fold {fold} - Loss & Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"fold_{fold}_metrics.png")
        plt.show()

        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        disp.plot(cmap='Blues', xticks_rotation='vertical')
        plt.title(f'Fold {fold} - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"fold_{fold}_confusion_matrix.png")
        plt.show()

        fold += 1

    print(f"\n✅ Average Macro F1-Score across {n_splits} folds: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
