import torchvision.transforms as transforms


class Transforms:
    def __init__(self):
        pass

    @staticmethod
    def normalize_scratch():
        return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    def transform_train(self):
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_scratch(),
            ]
        )

    def transform_test(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                self.normalize_scratch(),
            ]
        )
