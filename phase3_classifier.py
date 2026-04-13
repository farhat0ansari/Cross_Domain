import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Bidirectional, LSTM, Dense,
                                      Dropout, Embedding, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def build_bilstm_model(input_dim):
    """
    Build Bi-LSTM classifier
    Architecture from paper:
    - Embedding layer
    - Bidirectional LSTM with 32 units
    - 30% Dropout
    - Dense output layer
    """
    model = Sequential([
        # Reshape input for LSTM (treat embedding as sequence)
        tf.keras.layers.Reshape((1, input_dim), input_shape=(input_dim,)),
        # Bidirectional LSTM with 32 units
        Bidirectional(LSTM(32, return_sequences=False)),
        # 30% dropout
        Dropout(0.3),
        # Dense hidden layer
        Dense(64, activation='relu'),
        Dropout(0.2),
        # Output layer
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate(counterfactual_embeddings, source_labels,
                        target_reviews, target_labels,
                        n_folds=5, epochs=100, batch_size=32):
    """
    Phase III: Train Bi-LSTM on counterfactuals
    and evaluate on target domain (zero-shot)

    Uses 5-fold cross validation on training set
    as described in the paper
    """
    print("\nPhase III: Training Bi-LSTM Classifier")
    print("-" * 40)

    # Prepare data
    X_train = np.array(counterfactual_embeddings)
    y_train = np.array(source_labels)

    # Get target embeddings
    print("  Encoding target reviews with SBERT...")
    target_texts = [str(r) for r in target_reviews]
    X_test = sbert_model.encode(target_texts, show_progress_bar=False)
    X_test = normalize(X_test)
    y_test = np.array(target_labels)

    input_dim = X_train.shape[1]
    print(f"  Training set size: {len(X_train)}")
    print(f"  Test set size: {len(X_test)}")

    # 5-fold cross validation for hyperparameter validation
    print("  Running 5-fold cross validation...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]

        model = build_bilstm_model(input_dim)
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )

        val_preds = (model.predict(X_fold_val, verbose=0) > 0.5).astype(int).flatten()
        fold_acc = accuracy_score(y_fold_val, val_preds)
        cv_scores.append(fold_acc)
        print(f"    Fold {fold+1}: Validation Accuracy = {fold_acc:.4f}")

    print(f"  Average CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Train final model on all training data
    print("  Training final model on all counterfactuals...")
    final_model = build_bilstm_model(input_dim)
    early_stop = EarlyStopping(
        monitor='loss',
        patience=15,
        restore_best_weights=True
    )

    final_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    # Zero-shot prediction on target domain
    print("  Predicting on target domain (zero-shot)...")
    y_pred_prob = final_model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n  Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=['Real', 'Fake']))

    return accuracy, f1, final_model
