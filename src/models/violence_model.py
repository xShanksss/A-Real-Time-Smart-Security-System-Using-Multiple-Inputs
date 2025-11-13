"""
violence_detection_system/src/models/violence_detector.py
Hybrid CNN-LSTM model for violence detection - Improved Version
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import json
import gc
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure TensorFlow memory growth to prevent OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        logger.warning(f"GPU configuration error: {e}")


class ViolenceDetectionModel:
    def __init__(self, config_path='config/config.json'):
        """Initialize model with configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise

        self.sequence_length = self.config['model']['sequence_length']
        self.image_size = tuple(self.config['model']['image_size'])
        self.lstm_units = self.config['model']['lstm_units']
        self.model = None

    def build_model(self) -> Model:
        """Build hybrid CNN-LSTM architecture with error handling"""
        try:
            # Input shape: (sequence_length, height, width, channels)
            input_shape = (self.sequence_length, *self.image_size, 3)
            inputs = layers.Input(shape=input_shape)

            # CNN feature extractor (ResNet50 based)
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )

            # Freeze base model layers to reduce memory
            for layer in base_model.layers[:-20]:
                layer.trainable = False

            # TimeDistributed CNN with reduced complexity
            x = layers.TimeDistributed(base_model)(inputs)
            x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
            x = layers.TimeDistributed(layers.Dense(512, activation='relu'))(x)
            x = layers.TimeDistributed(layers.Dropout(0.5))(x)

            # LSTM layers for temporal analysis
            x = layers.LSTM(self.lstm_units, return_sequences=True)(x)
            x = layers.Dropout(0.5)(x)
            x = layers.LSTM(self.lstm_units // 2)(x)
            x = layers.Dropout(0.5)(x)

            # Dense layers
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.3)(x)

            # Output layer
            outputs = layers.Dense(1, activation='sigmoid')(x)

            self.model = Model(inputs=inputs, outputs=outputs)

            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()
                ]
            )

            logger.info("✓ Model built successfully")
            return self.model

        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: Optional[int] = None) -> keras.callbacks.History:
        """Train the model with automatic checkpoint resume and memory management"""

        if self.model is None:
            # If checkpoint exists, load it before building a new one
            checkpoint_path = Path('models/trained/violence_detector.h5')
            if checkpoint_path.exists():
                logger.info("Loading existing model checkpoint to resume training...")
                self.model = keras.models.load_model(checkpoint_path)
            else:
                self.build_model()

        if epochs is None:
            epochs = self.config['model']['epochs']

        batch_size = self.config['model']['batch_size']

        Path('models/trained').mkdir(parents=True, exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                'models/trained/violence_detector.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=False
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        try:
            tf.keras.backend.clear_session()
            gc.collect()

            logger.info("🚀 Starting / Resuming training...")

            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
                workers=4,
                use_multiprocessing=False,
                max_queue_size=10
            )

            gc.collect()
            return history

        except Exception as e:
            logger.error(f"Training error: {e}")
            tf.keras.backend.clear_session()
            gc.collect()
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance with error handling"""
        try:
            if self.model is None:
                raise ValueError("Model not built or loaded")

            results = self.model.evaluate(X_test, y_test, verbose=1)

            metrics = {
                'loss': float(results[0]),
                'accuracy': float(results[1]),
                'precision': float(results[2]),
                'recall': float(results[3]),
                'auc': float(results[4])
            }

            logger.info("\n" + "=" * 50)
            logger.info("MODEL EVALUATION RESULTS")
            logger.info("=" * 50)
            for metric, value in metrics.items():
                logger.info(f"{metric.capitalize()}: {value:.4f}")
            logger.info("=" * 50)

            return metrics

        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise

    def predict(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Make predictions with batch processing"""
        try:
            if self.model is None:
                raise ValueError("Model not built or loaded")

            if batch_size is None:
                batch_size = self.config['model'].get('batch_size', 32)

            predictions = self.model.predict(X, batch_size=batch_size, verbose=1)

            gc.collect()
            return predictions

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def save_model(self, path: str = 'models/trained/violence_detector.h5'):
        """Save trained model with error handling"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            logger.info(f"✓ Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: str = 'models/trained/violence_detector.h5') -> Model:
        """Load trained model with error handling"""
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            self.model = keras.models.load_model(path)
            logger.info(f"✓ Model loaded from {path}")
            return self.model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_summary(self):
        """Print model summary"""
        if self.model:
            return self.model.summary()
        else:
            logger.warning("Model not built yet")

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            tf.keras.backend.clear_session()
            gc.collect()
        except:
            pass


# Training script with robust error handling
def train_model() -> Tuple[ViolenceDetectionModel, keras.callbacks.History, Dict[str, float]]:
    """Main training function with comprehensive error handling"""

    try:
        logger.info("Loading data...")

        data_files = {
            'X_train': 'datasets/X_train.npy',
            'y_train': 'datasets/y_train.npy',
            'X_test': 'datasets/X_test.npy',
            'y_test': 'datasets/y_test.npy'
        }

        for name, path in data_files.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Data file not found: {path}")

        X_train = np.load('datasets/X_train.npy', mmap_mode='r')
        y_train = np.load('datasets/y_train.npy')
        X_test = np.load('datasets/X_test.npy', mmap_mode='r')
        y_test = np.load('datasets/y_test.npy')

        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")

        if len(X_train) != len(y_train):
            raise ValueError("Training data and labels have mismatched lengths")

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")

        detector = ViolenceDetectionModel()
        detector.build_model()
        detector.get_summary()

        logger.info("\nStarting training...")
        history = detector.train(X_train, y_train, X_val, y_val)

        logger.info("\nEvaluating on test set...")
        metrics = detector.evaluate(X_test, y_test)

        detector.save_model()

        import pickle
        history_path = 'models/trained/training_history.pkl'
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)

        logger.info("\n✓ Training completed successfully!")
        gc.collect()

        return detector, history, metrics

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        tf.keras.backend.clear_session()
        gc.collect()
        raise

    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        tf.keras.backend.clear_session()
        gc.collect()
        raise


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)