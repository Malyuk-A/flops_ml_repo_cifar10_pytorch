from typing import Any, Tuple

from flops_utils.ml_repo_building_blocks import load_dataset
from flops_utils.ml_repo_templates import DataManagerTemplate
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class DataManager(DataManagerTemplate):
    def __init__(self):
        self._prepare_data()

    def _prepare_data(self, partition_id=1) -> None:
        """Load partition CIFAR10 data."""
        dataset = load_dataset()
        dataset.set_format("numpy")

        # Divide data on each node: 80% train, 20% test
        dataset_train_test = dataset.train_test_split(test_size=0.2, seed=42)
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        def apply_transforms(batch):
            """Apply transforms to the partition from FederatedDataset."""
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        dataset_train_test = dataset_train_test.with_transform(apply_transforms)
        self.trainloader = DataLoader(
            dataset_train_test["train"], batch_size=32, shuffle=True
        )
        self.testloader = DataLoader(dataset_train_test["test"], batch_size=32)

    def get_data(self) -> Tuple[Any, Any]:
        return self.trainloader, self.testloader
