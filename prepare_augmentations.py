import ast
from torchvision import transforms


class PublicDataAugmentation(object):
    # Adopted from the original DINO implementation
    # Removed global_size and local_size
    def __init__(self, dataset_params):
        full_size = int(dataset_params['resolution'])
        global_size = int(dataset_params['resolution'])

        # Define the normalization
        normalize_mean = ast.literal_eval('(0.5071, 0.4867, 0.4408)')
        normalize_std = ast.literal_eval('(0.2675, 0.2565, 0.2761)')
        self.normalize = transforms.Compose([transforms.Normalize(normalize_mean, normalize_std),])

        # Define transforms for training (transforms_aug) and for validation (transforms_plain)
        self.transforms_plain = transforms.Compose([
            transforms.Resize(full_size, interpolation=3),
            transforms.CenterCrop(global_size),  # Center cropping
            transforms.ToTensor()])

        self.transforms_aug = self.transforms_plain

    def __call__(self, image):
        crops = []
        crops.append(self.transforms_aug(image))
        return crops
