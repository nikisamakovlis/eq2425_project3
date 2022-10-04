import ast
from torchvision import transforms


class PublicDataAugmentation(object):
    # Adopted from the original DINO implementation
    # Removed global_size and local_size
    def __init__(self, dataset_params):
        full_size = int(dataset_params['resolution'])
        global_size = int(dataset_params['resolution'])

        # Define the normalization
        self.normalize = transforms.Compose([transforms.Normalize(0.5, 1),])

        # Define transforms for training (transforms_aug) and for validation (transforms_plain)
        self.transforms_plain = transforms.Compose([
            transforms.Resize(full_size),
            transforms.CenterCrop(global_size),  # Center cropping
            transforms.ToTensor(),
            self.normalize])

        self.transforms_aug = self.transforms_plain

    def __call__(self, image):
        crops = []
        crops.append(self.transforms_aug(image))
        return crops
