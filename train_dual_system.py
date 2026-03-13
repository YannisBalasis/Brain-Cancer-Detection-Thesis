#!/usr/bin/env python3
"""
Training script for Dual Branch System
Implements 3-phase training strategy: Frozen ResNet -> Fine-tuning -> Full Optimization
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import logging

# Import our modules
from dual_system import DualBranchSystem, create_dual_system_model
from utils_dual import (
    DataLoader, ModelEvaluator, Visualizer, 
    save_results, create_directory_structure, 
    setup_logging, set_random_seeds
)
from config_dual import (
    get_config, get_paths_config, 
    TRAINING_CONFIG, CALLBACKS_CONFIG,
    REPRODUCIBILITY_CONFIG
)

class DualSystemTrainer:
    """
    Main trainer class for Dual Branch System
    """
    
    def __init__(self, data_path, config=None):
        """
        Initialize trainer
        
        Args:
            data_path: Path to dataset
            config: Configuration dictionary
        """
        self.data_path = data_path
        self.config = config if config else get_config()
        self.paths = get_paths_config(data_path)
        
        # Initialize components
        self.dual_system = None
        self.model = None
        self.data_loader = None
        self.evaluator = None
        self.visualizer = None
        
        # Training data
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.steps_info = None
        
        # Training history
        self.training_phases = {}
        self.total_training_time = 0
        
        # Create directory structure first
        create_directory_structure(self.paths['EXPERIMENT_DIR'])
        
        # Setup logging
        log_file = os.path.join(self.paths['LOGS_PATH'], 'training.log')
        setup_logging(log_file)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Dual System Trainer initialized")
        self.logger.info(f"Data path: {data_path}")
        self.logger.info(f"Experiment directory: {self.paths['EXPERIMENT_DIR']}")
    
    def setup_experiment(self):
        """
        Setup experiment environment
        """
        self.logger.info("Setting up experiment environment...")
        
        # Set random seeds for reproducibility
        if REPRODUCIBILITY_CONFIG['RANDOM_SEED']:
            set_random_seeds(REPRODUCIBILITY_CONFIG['RANDOM_SEED'])
        
        # Configure GPU if available
        self._configure_gpu()
        
        # Initialize components
        self.data_loader = DataLoader(self.data_path, self.config['data'])
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()
        
        self.logger.info("Experiment setup completed")
    
    def _configure_gpu(self):
        """
        Configure GPU settings
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"GPU configured successfully. Available GPUs: {len(gpus)}")
            except RuntimeError as e:
                self.logger.warning(f"GPU configuration failed: {e}")
        else:
            self.logger.info("No GPU found. Training will use CPU.")
    
    def load_data(self):
        """
        Load and prepare data
        """
        self.logger.info("Loading and preparing data...")
        
        # Create data generators
        self.train_generator, self.val_generator, self.test_generator, self.steps_info = \
            self.data_loader.create_data_generators(
                validation_split=self.config['data']['VALIDATION_SPLIT'],
                test_split=self.config['data']['TEST_SPLIT']
            )
        
        self.logger.info("Data loading completed")
        self.logger.info(f"Training samples: {self.steps_info['train_samples']}")
        self.logger.info(f"Validation samples: {self.steps_info['val_samples']}")
        self.logger.info(f"Test samples: {self.steps_info['test_samples']}")
    
    def build_model(self):
        """
        Build dual system model
        """
        self.logger.info("Building dual system model...")
        
        self.dual_system, self.model = create_dual_system_model()
        
        # Log model summary
        summary_info = self.dual_system.get_model_summary()
        self.logger.info("Model built successfully")
        self.logger.info(f"Total parameters: {summary_info['total_params']:,}")
        self.logger.info(f"Fusion strategy: {summary_info['fusion_strategy']}")
        
        # Save model summary to file
        summary_file = os.path.join(self.paths['RESULTS_PATH'], 'model_summary.txt')
        with open(summary_file, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        self.logger.info(f"Model summary saved to: {summary_file}")
    
    def _create_callbacks(self, phase, model_save_path):
        """
        Create training callbacks for specific phase
        
        Args:
            phase: Training phase name
            model_save_path: Path to save best model
            
        Returns:
            list: List of callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=CALLBACKS_CONFIG['EARLY_STOPPING']['MONITOR'],
            patience=CALLBACKS_CONFIG['EARLY_STOPPING']['PATIENCE'],
            restore_best_weights=CALLBACKS_CONFIG['EARLY_STOPPING']['RESTORE_BEST_WEIGHTS'],
            mode=CALLBACKS_CONFIG['EARLY_STOPPING']['MODE'],
            verbose=CALLBACKS_CONFIG['EARLY_STOPPING']['VERBOSE']
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor=CALLBACKS_CONFIG['REDUCE_LR']['MONITOR'],
            factor=CALLBACKS_CONFIG['REDUCE_LR']['FACTOR'],
            patience=CALLBACKS_CONFIG['REDUCE_LR']['PATIENCE'],
            min_lr=CALLBACKS_CONFIG['REDUCE_LR']['MIN_LR'],
            mode=CALLBACKS_CONFIG['REDUCE_LR']['MODE'],
            verbose=CALLBACKS_CONFIG['REDUCE_LR']['VERBOSE']
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            filepath=model_save_path,
            monitor=CALLBACKS_CONFIG['MODEL_CHECKPOINT']['MONITOR'],
            save_best_only=CALLBACKS_CONFIG['MODEL_CHECKPOINT']['SAVE_BEST_ONLY'],
            save_weights_only=CALLBACKS_CONFIG['MODEL_CHECKPOINT']['SAVE_WEIGHTS_ONLY'],
            mode=CALLBACKS_CONFIG['MODEL_CHECKPOINT']['MODE'],
            verbose=CALLBACKS_CONFIG['MODEL_CHECKPOINT']['VERBOSE']
        )
        callbacks.append(checkpoint)
        
        self.logger.info(f"Callbacks created for {phase}")
        return callbacks
    
    def train_phase1(self):
        """
        Phase 1: Train with frozen ResNet-50
        """
        phase_name = "Phase 1: Frozen ResNet"
        self.logger.info(f"Starting {phase_name}")
        
        # Ensure ResNet is frozen
        self.dual_system.freeze_resnet_layers()
        
        # Compile model for phase 1
        self.dual_system.compile_model('phase1')
        
        # Setup callbacks
        model_save_path = os.path.join(self.paths['MODEL_SAVE_PATH'], self.paths['PHASE1_MODEL'])
        callbacks = self._create_callbacks('phase1', model_save_path)
        
        # Training configuration
        phase1_config = TRAINING_CONFIG['PHASE1']
        
        # Train
        start_time = time.time()
        
        history = self.model.fit(
            self.train_generator,
            epochs=phase1_config['EPOCHS'],
            validation_data=self.val_generator,
            steps_per_epoch=self.steps_info['train_steps'],
            validation_steps=self.steps_info['val_steps'],
            callbacks=callbacks,
            verbose=1
        )
        
        phase1_time = time.time() - start_time
        
        # Store training history
        self.training_phases['phase1'] = {
            'history': history.history,
            'training_time': phase1_time,
            'config': phase1_config
        }
        
        # Log phase completion
        best_val_acc = max(history.history['val_accuracy'])
        self.logger.info(f"{phase_name} completed in {phase1_time/60:.2f} minutes")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Save phase history
        history_file = os.path.join(self.paths['RESULTS_PATH'], 'phase1_history.json')
        with open(history_file, 'w') as f:
            json.dump(history.history, f, indent=2)
    
    def train_phase2(self):
        """
        Phase 2: Fine-tune ResNet-50 layers
        """
        phase_name = "Phase 2: ResNet Fine-tuning"
        self.logger.info(f"Starting {phase_name}")
        
        # Unfreeze ResNet layers
        unfreeze_from = self.config['model']['RESNET50']['UNFREEZE_FROM_LAYER']
        self.dual_system.unfreeze_resnet_layers(unfreeze_from)
        
        # Compile model for phase 2 (lower learning rate)
        self.dual_system.compile_model('phase2')
        
        # Setup callbacks
        model_save_path = os.path.join(self.paths['MODEL_SAVE_PATH'], self.paths['PHASE2_MODEL'])
        callbacks = self._create_callbacks('phase2', model_save_path)
        
        # Training configuration
        phase2_config = TRAINING_CONFIG['PHASE2']
        
        # Train
        start_time = time.time()
        
        history = self.model.fit(
            self.train_generator,
            epochs=phase2_config['EPOCHS'],
            validation_data=self.val_generator,
            steps_per_epoch=self.steps_info['train_steps'],
            validation_steps=self.steps_info['val_steps'],
            callbacks=callbacks,
            verbose=1
        )
        
        phase2_time = time.time() - start_time
        
        # Store training history
        self.training_phases['phase2'] = {
            'history': history.history,
            'training_time': phase2_time,
            'config': phase2_config
        }
        
        # Log phase completion
        best_val_acc = max(history.history['val_accuracy'])
        self.logger.info(f"{phase_name} completed in {phase2_time/60:.2f} minutes")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Save phase history
        history_file = os.path.join(self.paths['RESULTS_PATH'], 'phase2_history.json')
        with open(history_file, 'w') as f:
            json.dump(history.history, f, indent=2)
    
    def train_phase3(self, enable_phase3=True):
        """
        Phase 3: Full model optimization (optional)
        
        Args:
            enable_phase3: Whether to run phase 3
        """
        if not enable_phase3:
            self.logger.info("Phase 3 skipped")
            return
        
        phase_name = "Phase 3: Full Optimization"
        self.logger.info(f"Starting {phase_name}")
        
        # All layers should already be unfrozen from phase 2
        # Compile model for phase 3 (very low learning rate)
        self.dual_system.compile_model('phase3')
        
        # Setup callbacks
        model_save_path = os.path.join(self.paths['MODEL_SAVE_PATH'], self.paths['PHASE3_MODEL'])
        callbacks = self._create_callbacks('phase3', model_save_path)
        
        # Training configuration
        phase3_config = TRAINING_CONFIG['PHASE3']
        
        # Train
        start_time = time.time()
        
        history = self.model.fit(
            self.train_generator,
            epochs=phase3_config['EPOCHS'],
            validation_data=self.val_generator,
            steps_per_epoch=self.steps_info['train_steps'],
            validation_steps=self.steps_info['val_steps'],
            callbacks=callbacks,
            verbose=1
        )
        
        phase3_time = time.time() - start_time
        
        # Store training history
        self.training_phases['phase3'] = {
            'history': history.history,
            'training_time': phase3_time,
            'config': phase3_config
        }
        
        # Log phase completion
        best_val_acc = max(history.history['val_accuracy'])
        self.logger.info(f"{phase_name} completed in {phase3_time/60:.2f} minutes")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Save phase history
        history_file = os.path.join(self.paths['RESULTS_PATH'], 'phase3_history.json')
        with open(history_file, 'w') as f:
            json.dump(history.history, f, indent=2)
    
    def train_complete_system(self, enable_phase3=True):
        """
        Train the complete dual system through all phases
        
        Args:
            enable_phase3: Whether to run the optional phase 3
        """
        self.logger.info("Starting complete dual system training")
        total_start_time = time.time()
        
        # Phase 1: Frozen ResNet
        self.train_phase1()
        
        # Phase 2: ResNet Fine-tuning
        self.train_phase2()
        
        # Phase 3: Full optimization (optional)
        self.train_phase3(enable_phase3)
        
        # Calculate total training time
        self.total_training_time = time.time() - total_start_time
        
        # Save best model
        self._save_best_model()
        
        # Generate training visualizations
        self._create_training_visualizations()
        
        # Save complete training summary
        self._save_training_summary()
        
        self.logger.info("Complete training finished")
        self.logger.info(f"Total training time: {self.total_training_time/3600:.2f} hours")
    
    def _save_best_model(self):
        """
        Save the best performing model
        """
        # Determine best phase based on validation accuracy
        best_phase = None
        best_accuracy = 0
        
        for phase, data in self.training_phases.items():
            phase_best_acc = max(data['history']['val_accuracy'])
            if phase_best_acc > best_accuracy:
                best_accuracy = phase_best_acc
                best_phase = phase
        
        # Load and save best model
        if best_phase:
            phase_model_file = self.paths[f'{best_phase.upper()}_MODEL']
            source_path = os.path.join(self.paths['MODEL_SAVE_PATH'], phase_model_file)
            best_path = os.path.join(self.paths['MODEL_SAVE_PATH'], self.paths['BEST_MODEL'])
            
            if os.path.exists(source_path):
                # Load the best model
                self.model = tf.keras.models.load_model(source_path)
                # Save as best model
                self.model.save(best_path)
                
                self.logger.info(f"Best model from {best_phase} saved")
                self.logger.info(f"Best validation accuracy: {best_accuracy:.4f}")
            else:
                self.logger.warning(f"Best model file not found: {source_path}")
    
    def _create_training_visualizations(self):
        """
        Create and save training visualizations
        """
        self.logger.info("Creating training visualizations...")
        
        # Combined training history plot
        combined_history = {}
        
        # Combine all phases
        for phase, data in self.training_phases.items():
            for metric, values in data['history'].items():
                if metric not in combined_history:
                    combined_history[metric] = []
                combined_history[metric].extend(values)
        
        # Plot combined history
        plot_path = os.path.join(self.paths['PLOTS_PATH'], 'training_history.png')
        self.visualizer.plot_training_history(combined_history, plot_path)
        
        # Individual phase plots
        for phase, data in self.training_phases.items():
            phase_plot_path = os.path.join(self.paths['PLOTS_PATH'], f'{phase}_history.png')
            self.visualizer.plot_training_history(data['history'], phase_plot_path)
        
        self.logger.info("Training visualizations created")
    
    def _save_training_summary(self):
        """
        Save complete training summary
        """
        summary = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'data_path': self.data_path,
                'experiment_dir': self.paths['EXPERIMENT_DIR'],
                'total_training_time_hours': self.total_training_time / 3600
            },
            'data_info': self.steps_info,
            'model_info': self.dual_system.get_model_summary(),
            'training_phases': {}
        }
        
        # Add phase information (without full history for brevity)
        for phase, data in self.training_phases.items():
            summary['training_phases'][phase] = {
                'training_time_minutes': data['training_time'] / 60,
                'epochs': len(data['history']['accuracy']),
                'best_val_accuracy': max(data['history']['val_accuracy']),
                'final_val_accuracy': data['history']['val_accuracy'][-1],
                'config': data['config']
            }
        
        # Save summary
        summary_file = os.path.join(self.paths['RESULTS_PATH'], 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Training summary saved to: {summary_file}")

def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description='Train Dual Branch System')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--phase3', action='store_true',
                       help='Enable optional phase 3 training')
    parser.add_argument('--config', type=str,
                       help='Path to custom configuration file')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("DUAL BRANCH SYSTEM TRAINING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.data}")
    print(f"Phase 3 enabled: {args.phase3}")
    print("=" * 80)
    
    try:
        # Initialize trainer
        trainer = DualSystemTrainer(args.data)
        
        # Setup experiment
        trainer.setup_experiment()
        
        # Load data
        trainer.load_data()
        
        # Build model
        trainer.build_model()
        
        # Train complete system
        trainer.train_complete_system(enable_phase3=args.phase3)
        
        print("=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Experiment directory: {trainer.paths['EXPERIMENT_DIR']}")
        print(f"Total training time: {trainer.total_training_time/3600:.2f} hours")
        
        # Print best results
        best_accuracies = []
        for phase, data in trainer.training_phases.items():
            best_acc = max(data['history']['val_accuracy'])
            best_accuracies.append((phase, best_acc))
        
        best_phase, best_acc = max(best_accuracies, key=lambda x: x[1])
        print(f"Best performance: {best_acc:.4f} from {best_phase}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Training failed: {e}")
        logging.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()