from abc import abstractmethod

from tools.run.config_runner import ConfigRunner


class TrainableRunner(ConfigRunner):
    """Trainable Runner which can also train."""

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        pass

    def load(self, *args, **kwargs) -> None:
        """Loading a trainable runner by a given state.
        """
        pass
