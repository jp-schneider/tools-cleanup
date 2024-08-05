
from typing import Callable, List, Optional
from tools.dataset.base_dataset import BaseDataset
from tools.io.image import load_image
from tools.util.path_tools import format_os_independent, read_directory


class ImagePathDataset(BaseDataset):

    _paths: List[str]
    """List of paths to the images."""

    _load_image_args: dict
    """Arguments for the load_image function."""

    _transform: Optional[Callable]
    """Transform function for the images."""

    def __init__(self,
                 paths: List[str],
                 transform: Optional[Callable] = None,
                 load_image_args: Optional[dict] = None) -> None:
        self._paths = paths
        self._transform = transform
        self._load_image_args = load_image_args

    def __getitem__(self, index: int) -> str:
        path = self._paths[index]
        args = self._load_image_args or {}
        image = load_image(path, args)
        if self._transform:
            image = self._transform(image)
        return image

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def paths(self) -> List[str]:
        """Returns the paths to the images."""
        return list(self._paths)

    @classmethod
    def from_folder(
        cls,
        path: str,
        filename_format: str = r"(?P<index>[0-9]+).png",
        **kwargs
    ) -> 'ImagePathDataset':
        """Creates a dataset from a folder of images.

        Parameters
        ----------
        path : str
            Path of the folder containing the images.
        filename_format : str, optional
            How the image files are formated, regex to match to be included, by default r"(?P<index>[0-9]+).png"
            Images will be sorted by the index.

        kwargs
            Additional arguments for the dataset constructor.

        Returns
        -------
        ImagePathDataset
            Created dataset.
        """
        image_paths = read_directory(
            path, filename_format, parser=dict(index=int), path_key="path")
        sorted_index = sorted(image_paths, key=lambda x: x["index"])
        image_paths = [format_os_independent(x["path"]) for x in sorted_index]
        return cls(image_paths, **kwargs)
