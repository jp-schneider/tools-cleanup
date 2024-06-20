from typing import Iterable, Optional, Set, FrozenSet

from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.typing import VEC_TYPE
from abc import abstractmethod
from tools.logger.logging import logger
from tools.logger.prefix_logger_adapter import PrefixLoggerAdapter
from logging import Logger


class SceneNode(AbstractSceneNode):
    """Scene not class for nodes representing a geometrical scene / coordinate system."""

    _name: Optional[str]
    """Name if the scene node for identification."""

    _parent: Optional['AbstractSceneNode']
    """Parent of this node. If None, this node is the root of the scene."""

    _scene_children: Set['AbstractSceneNode']
    """Children of this node."""

    logger: Logger
    """Logger for this scene node."""

    def __init__(self,
                 name: Optional[str] = None,
                 children: Optional[Iterable['AbstractSceneNode']] = None,
                 decoding: bool = False,
                 **kwargs
                 ) -> None:
        """Abstract class for nodes within a geometrical scene.

        Parameters
        ----------
        name : Optional[str], optional
            The name of the node for display purposes, by default None
        children : Optional[Iterable[&#39;AbstractSceneNode&#39;]], optional
            Its node children, by default None
        decoding : bool, optional
            If its currently beeing decoded and checks should be ommited, by default False
        """
        super().__init__(decoding=decoding, **kwargs)
        self._parent = None
        self._name = name
        self._scene_children = set()
        if children is not None:
            self.add_scene_children(*children)
        _logger = logger.getChild(type(self).__name__ + f"_{id(self)}")
        if name is not None and len(name) > 0:
            _logger = PrefixLoggerAdapter(_logger, dict(prefix=name))
        self.logger = _logger

    def __ignore_on_iter__(self) -> Set[str]:
        ret = super().__ignore_on_iter__()
        ret.add('_parent')
        return ret

    def after_decoding(self, **kwargs) -> None:
        super().after_decoding(**kwargs)
        # Set parent of children
        for child in self._scene_children:
            child.set_parent(self)

    def get_name(self) -> Optional[str]:
        """Get the name of the node.

        Returns
        -------
        Optional[str]
            The name of the node. If no name is set, returns None.
        """
        return self._name

    def set_name(self, value: Optional[str]) -> None:
        """Sets the name of the node.

        Parameters
        ----------
        value : Optional[str]
            New name of the node or None to remove the name.
        """
        self._name = value

    def add_scene_children(self, *children: 'AbstractSceneNode', **kwargs) -> None:
        """
        Add children to scene node.
        Will set the parent of the children to this node, before adding them.

        Parameters
        ----------
        children : AbstractSceneNode
            Children to add.
        """
        for child in children:
            child.set_parent(self)
            self._scene_children.add(child)

    def remove_scene_children(self, *children: 'AbstractSceneNode', **kwargs) -> Set["AbstractSceneNode"]:
        """
        Remove children from scene node.
        Removes the parent of the children before removing them as children.

        Parameters
        ----------
        children : AbstractSceneNode
            Children to remove.

        Returns
        -------
        Set[AbstractSceneNode]
            The set of children that were actually removed.
        """
        ret = set()
        for child in children:
            if child not in self._scene_children:
                continue
            child.set_parent(None)
            self._scene_children.remove(child)
            ret.add(child)
        return ret

    def get_scene_children(self) -> FrozenSet["AbstractSceneNode"]:
        return frozenset(self._scene_children)

    def set_parent(self, parent: Optional['AbstractSceneNode']) -> None:
        """Set the parent of the node.

        Parameters
        ----------
        parent : AbstractSceneNode
            The new parent of the node.
        """
        self._parent = parent

    def get_parent(self) -> Optional['AbstractSceneNode']:
        """Get the parent of the node.

        Returns
        -------
        Optional[AbstractSceneNode]
            The parent of the node. If the node has no parent, returns None.
        """
        return self._parent
