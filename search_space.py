import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import json
from typing import List

class HPOptimization:
    def __init__(
        self,
        lr: float,
        epochs: int,
        exp_dir: str,
    ):
        self.lr = lr
        self.epochs = epochs
        self.exp_dir = exp_dir
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def _evaluate(self, model, dataloader):
        """
        Evaluate the model accuracy on the provided dataloader.
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy

    def _get_mnist_dataset(self, batch_size=128):
        """
        Returns MNIST dataset for training.
        """
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # LeNet-5 expects 32x32 input
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        # Create validation set as holdout from training set
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        return train_loader, val_loader, test_loader

    def _search_space_by_func(self, trial):
        """
        Define-by-run function to create the search space for LeNet-5.
        """
        # 1. Search for number of filters in Convolutional Layers
        # LeNet original: C1=6, C3=16. We search around these values.
        trial.suggest_int(name="c1_filters", low=4, high=16, step=2)
        trial.suggest_int(name="c3_filters", low=10, high=30, step=2)

        # 2. Search for number of units in Fully Connected Layers
        # LeNet original: F5=120, F6=84.
        trial.suggest_int(name="f5_units", low=80, high=150, step=10)
        trial.suggest_int(name="f6_units", low=60, high=100, step=5)

        # 3. Search for activation functions
        # Comparing standard Tanh (original) vs ReLU (modern)
        trial.suggest_categorical(
            name="activation",
            choices=["tanh", "relu"],
        )

        return trial.params

    def _create_model(self, trial_hp):
        """Create LeNet-5 model based on hyperparameters"""
        
        # Define the dynamic class inside to capture the parameters easily
        class LeNet5(nn.Module):
            def __init__(self, hp):
                super(LeNet5, self).__init__()
                
                # Select Activation
                if hp['activation'] == "tanh":
                    self.act = nn.Tanh()
                else:
                    self.act = nn.ReLU()

                # Layer C1: Input 1 channel -> Output c1_filters
                # Input: 32x32 -> Conv(5x5) -> 28x28 -> AvgPool(2x2) -> 14x14
                self.c1 = nn.Conv2d(1, hp['c1_filters'], kernel_size=5)
                self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
                
                # Layer C3: Input c1_filters -> Output c3_filters
                # Input: 14x14 -> Conv(5x5) -> 10x10 -> AvgPool(2x2) -> 5x5
                self.c3 = nn.Conv2d(hp['c1_filters'], hp['c3_filters'], kernel_size=5)
                self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
                
                # Flatten size calculation:
                # The spatial dimension at this point is 5x5.
                self.flatten_size = hp['c3_filters'] * 5 * 5
                
                # Layer C5/F5: Fully Connected
                self.f5 = nn.Linear(self.flatten_size, hp['f5_units'])
                
                # Layer F6: Fully Connected
                self.f6 = nn.Linear(hp['f5_units'], hp['f6_units'])
                
                # Output Layer: 10 digits
                self.output = nn.Linear(hp['f6_units'], 10)

            def forward(self, x):
                # C1 -> Act -> S2
                x = self.s2(self.act(self.c1(x)))
                
                # C3 -> Act -> S4
                x = self.s4(self.act(self.c3(x)))
                
                # Flatten
                x = x.view(-1, self.flatten_size)
                
                # F5 -> Act
                x = self.act(self.f5(x))
                
                # F6 -> Act
                x = self.act(self.f6(x))
                
                # Output (No activation, CrossEntropyLoss handles Softmax)
                x = self.output(x)
                return x

        # Return the instantiated model moved to the correct device
        model = LeNet5(trial_hp)
        return model.to(self.device)

    def _train_model(self, model, train_loader, val_loader, epochs, lr):
        """Train the PyTorch model"""
        
        criterion = nn.CrossEntropyLoss()
        # Using SGD as per original LeNet spirit, but standard optimization
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        best_val_acc = 0.0

        for epoch in range(epochs):
            # --- Training Phase ---
            model.train()
            running_loss = 0.0
            
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            # --- Validation Phase ---
            val_acc = self._evaluate(model, val_loader)
            
            # Save checkpoint if this is the best model so far
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(self.work_dir, "best_model.pth"))
        
        return best_val_acc

    def __call__(
        self,
        trial,
    ):
        try:
            # create work directory for storing trial artifacts
            self.work_dir = os.path.join(
                self.exp_dir,
                f"trial_{trial.number}",
            )

            os.makedirs(self.work_dir, exist_ok=True)

            # get trial hyperparameters
            self.trial_hp = self._search_space_by_func(trial)
            self._log_trial_hps()

            # instantiate model
            model = self._create_model(self.trial_hp)

            # get MNIST dataset for training
            train_loader, val_loader, test_loader = self._get_mnist_dataset()

            # train model
            val_accuracy = self._train_model(
                model,
                train_loader,
                val_loader,
                self.epochs,
                self.lr
            )

            # log results
            self.val_accuracy = val_accuracy

            self._log_trial_status(status=True)
        except Exception as e:
            print(f"Trial failed: {e}") # Added simple print for debugging
            self._log_trial_status(status=False)
            return 0.0 # Return 0 accuracy on failure

        # report metrics
        return self.val_accuracy  # Changed from [self.val_accuracy] to match standard Optuna float return

    def _log_trial_status(self, status):
        """
        Writes a json file indicating the status of the trial.
        """
        with open(
            os.path.join(self.work_dir, "trial_status.json"), "w"
        ) as outfile:
            json.dump(
                {'trial_finished_successfully': status}, outfile, indent=1
            )

    def _log_trial_hps(self):
        """
        Writes a json file indicating the status of the trial.
        """
        with open(
            os.path.join(self.work_dir, "trial_hps.json"), "w"
        ) as outfile:
            json.dump(self.trial_hp, outfile, indent=1)