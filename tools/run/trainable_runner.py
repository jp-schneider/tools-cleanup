from abc import abstractmethod

from tools.run.config_runner import ConfigRunner



class TrainableRunner(ConfigRunner):
    """Trainable Runner which can also train."""

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        pass
