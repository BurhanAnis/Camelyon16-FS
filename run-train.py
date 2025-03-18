import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import numpy as np
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from train import HistologyTileDataset, create_dataloaders

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    """Calculate various classification metrics"""
    # Convert tensors to numpy arrays
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Calculate sensitivity (same as recall) and specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = recall  # Same as recall
    specificity = tn / (tn + fp)
    
    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1
    }

# Function to save the best model
class SaveBestModel:
    """Save the best model based on validation metrics"""
    def __init__(self, best_valid_metric=0.0, metric_name='f1', save_dir='models'):
        self.best_valid_metric = best_valid_metric
        self.metric_name = metric_name
        self.save_dir = save_dir
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def __call__(self, current_valid_metric, epoch, model, optimizer, metrics):
        if current_valid_metric > self.best_valid_metric:
            self.best_valid_metric = current_valid_metric
            print(f"\nBest validation {self.metric_name}: {self.best_valid_metric:.4f}")
            print(f"Saving best model for epoch: {epoch+1}\n")
            
            # Save the model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, f'{self.save_dir}/best_model.pth')

# Training function with tqdm monitoring
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, validate_every=5):
    since = time.time()
    
    # Initialize metrics tracking
    best_model_wts = model.state_dict()
    best_f1 = 0.0
    
    # Metrics history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training on device: {device}")
    print(f"Training set size: {len(train_loader.dataset)} samples")
    print(f"Validation set size: {len(val_loader.dataset)} samples")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Total batches per epoch: {len(train_loader)}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        # Iterate over data with tqdm progress bar
        print(f"Training phase - Epoch {epoch+1}/{num_epochs}")
        train_pbar = tqdm(train_loader, desc=f"Training", leave=True)
        batch_count = 0
        
        for inputs, labels in train_pbar:
            batch_count += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass + optimize
                loss.backward()
                optimizer.step()
            
            # Statistics
            batch_loss = loss.item() * inputs.size(0)
            running_loss += batch_loss
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            # Update progress bar with current batch loss
            train_pbar.set_postfix({
                'batch': f'{batch_count}/{len(train_loader)}',
                'loss': f'{batch_loss/inputs.size(0):.4f}'
            })
        
        # Calculate epoch loss and metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        train_metrics = calculate_metrics(torch.tensor(all_labels), torch.tensor(all_preds))
        
        # Update history
        history['train_loss'].append(epoch_loss)
        history['train_metrics'].append(train_metrics)
        
        # Print training metrics
        print(f'Train Loss: {epoch_loss:.4f}')
        print(f'Train Metrics:')
        for metric_name, metric_value in train_metrics.items():
            print(f'  {metric_name}: {metric_value:.4f}')
        
        # Validation phase (every validate_every epochs)
        if (epoch + 1) % validate_every == 0 or epoch == num_epochs - 1:
            print(f"Starting validation at epoch {epoch+1}")
            model.eval()
            running_loss = 0.0
            all_labels = []
            all_preds = []
            
            # Iterate over validation data with tqdm
            val_pbar = tqdm(val_loader, desc=f"Validating", leave=True)
            
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    batch_loss = loss.item() * inputs.size(0)
                    running_loss += batch_loss
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    
                    # Update progress bar
                    val_pbar.set_postfix({'loss': f'{batch_loss/inputs.size(0):.4f}'})
            
            # Calculate validation loss and metrics
            val_loss = running_loss / len(val_loader.dataset)
            val_metrics = calculate_metrics(torch.tensor(all_labels), torch.tensor(all_preds))
            
            # Update history
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            # Print validation metrics
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Metrics:')
            for metric_name, metric_value in val_metrics.items():
                print(f'  {metric_name}: {metric_value:.4f}')
            
            # Save the best model
            save_best_model(val_metrics['f1'], epoch, model, optimizer, val_metrics)
            
            # Check if this is the best model based on F1 score
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_model_wts = model.state_dict()
                print(f"New best model found! F1: {best_f1:.4f}")
        
        # Step the scheduler
        scheduler.step()
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'histology_models/checkpoint_epoch_{epoch+1}.pth'
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'metrics': train_metrics,
            }, checkpoint_path)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time//60:.0f}m {epoch_time%60:.0f}s")
        print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val F1: {best_f1:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

# Function to visualize results
def plot_training_history(history):
    """Plot training and validation metrics"""
    import matplotlib.pyplot as plt
    
    # Plot training & validation loss
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'])
    plt.plot(range(0, len(history['val_loss'])*5, 5), history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot metrics
    metrics = ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        
        # Extract metric values
        train_values = [m[metric] for m in history['train_metrics']]
        val_values = [m[metric] for m in history['val_metrics']]
        
        plt.plot(train_values)
        plt.plot(range(0, len(val_values)*5, 5), val_values)
        plt.title(f'Model {metric.capitalize()}')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig('histology_models/training_history.png')
    plt.show()

# Wrap the main execution code in if __name__ == '__main__': block
if __name__ == '__main__':
    # Define paths
    pos_slide_path = '/Volumes/BurhanAnisExtDrive/camelyon/camelyon_data/training/positive/images'
    neg_slide_path = '/Volumes/BurhanAnisExtDrive/camelyon/camelyon_data/training/negative'
    pos_grid_path = '/Users/burhananis/fully-supervised-camelyon/data/tumour_grid.pkl'
    neg_grid_path = '/Users/burhananis/fully-supervised-camelyon/data/tile_coords_neg.pkl'
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        pos_slide_path=pos_slide_path,
        neg_slide_path=neg_slide_path,
        pos_grid_path=pos_grid_path,
        neg_grid_path=neg_grid_path,
        batch_size=8,
        tile_size=256,
        samples_per_class=1000,
        num_workers=4
    )
    
    # Set device
    device = torch.device("mps")
    
    # Load pretrained model (ResNet50)
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # Modify the final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification: tumor or not tumor
    
    # Move model to device
    model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Initialize SaveBestModel
    save_best_model = SaveBestModel(metric_name='f1', save_dir='histology_models')
    
    # Train the model
    model, history = train_model(model, criterion, optimizer, scheduler,
                               num_epochs=25, validate_every=5)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, 'histology_models/final_model.pth')
    
    # Plot the training history
    plot_training_history(history)

