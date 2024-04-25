import warnings
from collections import OrderedDict

import torch
from tqdm import tqdm

from data_manager import DataManager
from model_manager import ModelManager

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Client:
    def __init__(self):
        self.model = ModelManager().get_model()
        self.trainloader, self.testloader = DataManager().get_data()

    def get_model_parameters(self):
        # return self.model.get_weights()
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_model_parameters(self, parameters):
        # self.model.set_weights(parameters)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit_model(self):
        # self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        # return len(self.x_train)

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

        return self.trainloader.dataset

    def evaluate_model(self):
        # loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        # return loss, accuracy, len(self.x_test)

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
        return loss, accuracy, self.testloader.dataset
