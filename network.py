import numpy as np
from nn_library.utils.batch_generator import BatchGenerator

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss_function):
        self.loss = loss_function

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, optimizer, batch_size=32, validation_data=None):
        train_loss_history = []
        val_loss_history = []

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_loss = 0
            batch_generator = BatchGenerator(x_train, y_train, batch_size)
            
            for x_batch, y_batch in batch_generator:
                # 1. Forward pass
                output = self.predict(x_batch)
                
                # 2. Compute loss
                loss = self.loss.forward(output, y_batch)
                epoch_loss += loss

                # 3. Backward pass
                gradient = self.loss.backward(output, y_batch)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient)
                
                # 4. Update weights
                for layer in self.layers:
                    optimizer.update(layer)
            
            # --- Store training loss for the epoch ---
            train_loss_history.append(epoch_loss / len(batch_generator))

            # --- Validation Step (at the end of each epoch) ---
            if validation_data is not None:
                X_val, y_val = validation_data
                val_output = self.predict(X_val)
                val_loss = self.loss.forward(val_output, y_val)
                val_loss_history.append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_history[-1]:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_history[-1]:.6f}")

        return train_loss_history, val_loss_history
