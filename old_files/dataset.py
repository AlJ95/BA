from torch.utils.data.dataset import Dataset as DatasetClass
from createImages_cv2 import produce_image


class Dataset(DatasetClass):

    def __init__(self, plot, no_of_images, salt_and_pepper_prob=0.05, dim=(1080, 1920, 4)):
        self.plot = plot
        self.no_of_images = no_of_images
        self.salt_and_pepper_prob = salt_and_pepper_prob
        self.dim = dim
        pass

    def __getitem__(self, index):
        return produce_image(self.plot, self.no_of_images, self.salt_and_pepper_prob, self.dim)

    def __len__(self):
        return 10000
