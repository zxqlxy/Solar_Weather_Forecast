from torchvision.datasets.folder import DatasetFolder
import numpy as np

EXTENSIONS = ('.fits', '.npz', '.npy')

def default_loader(path: str) -> Any:
    with open(path, 'rb') as f:
        src = np.load(path)
        src[src <= 1] = 1
        # src = np.log(src)
        return src

class ImageFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples