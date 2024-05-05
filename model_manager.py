import warnings
from collections import OrderedDict
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_manager import DataManager
from flops_utils.ml_repo_templates import ModelManagerTemplate
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ModelManager(ModelManagerTemplate):
    def __init__(self):
        self.model = Net().to(DEVICE)

    def prepare_data(self) -> None:
        self.trainloader, self.testloader = DataManager().get_data()

    def get_model(self) -> Any:
        return self.model

    def get_model_parameters(self) -> Any:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_model_parameters(self, parameters) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit_model(self) -> int:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        epochs = 1  # todo
        for _ in range(epochs):
            for batch in tqdm(self.trainloader, "Training"):
                images = batch["img"]
                labels = batch["label"]
                optimizer.zero_grad()
                criterion(self.model(images.to(DEVICE)), labels.to(DEVICE)).backward()
                optimizer.step()

        return len(self.trainloader.dataset)

    def evaluate_model(self) -> Tuple[Any, Any, int]:
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for batch in tqdm(self.testloader, "Testing"):
                images = batch["img"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(self.testloader.dataset)
        return loss, accuracy, len(self.testloader.dataset)
