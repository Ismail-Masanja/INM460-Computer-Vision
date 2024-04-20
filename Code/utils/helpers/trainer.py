import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


__all__ = ['Trainer']


class Trainer:
    """
    Trains pytorch models. After init, call train passing the model to train.

    REF: Ismail Masanja N3063/INM702: Mathematics and Programming for AI. Heavy modification 
        done to fit current project.
    """

    def __init__(self, training_data, testing_data, optimizer, costfx,
                 device, epochs=10, scheduler=None):
        self.trnd = training_data
        self.tstd = testing_data
        self.epochs = epochs
        self.optim = optimizer
        self.costfx = costfx
        self.device = device
        self.scheduler = scheduler

    def train(self, model):
        model.to(self.device)
        training_losses = []
        validation_losses = []

        for epoch in tqdm(range(self.epochs), desc='Training Epochs'):
            model.train()
            running_loss = 0.0
            for inputs, targets in self.trnd:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                # Zero the parameter gradients
                self.optim.zero_grad()

                # Forward + backward + optimize
                outputs = model(inputs)
                loss = self.costfx(outputs, targets)
                loss.backward()
                self.optim.step()

                running_loss += loss.item()

            if self.scheduler is not None:
                self.scheduler.step()

            # Calculate and store the average loss for this epoch
            avg_train_loss = running_loss / len(self.trnd)
            training_losses.append(avg_train_loss)

            # Validation loss
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.tstd:
                    inputs, targets = inputs.to(
                        self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = self.costfx(outputs, targets)
                    running_loss += loss.item()

            avg_test_loss = running_loss / len(self.tstd)
            validation_losses.append(avg_test_loss)

            print(
                f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")

        # Plotting the training and validation loss
        self._plot_losses(training_losses, validation_losses)

    def _plot_losses(self, training_losses, validation_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs+1),
                 training_losses, label='Training Loss')
        plt.plot(range(1, self.epochs+1),
                 validation_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # Ensure x-axis labels are whole numbers
        plt.xticks(np.arange(1, self.epochs + 1, 1.0))
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.show()
