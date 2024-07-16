from typing import TYPE_CHECKING, FrozenSet, Iterable, Optional, Set

from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.typing import VEC_TYPE


class ModuleSceneParent(AbstractSceneNode):
    """Parent class for scene nodes that wrap a module class.
    Needed, as a pytorch module object can not deal with cyclic references.
    """

    _node: 'AbstractSceneNode'
    """The actual node that is wrapped by this parent wrapper class."""

    def __init__(self, node: 'AbstractSceneNode', **kwargs):
        super().__init__(**kwargs)
        if node is None:
            raise ValueError("Node must not be None.")
        self._node = node

    def get_name(self) -> Optional[str]:
        return self._node.get_name()

    def set_name(self, value: Optional[str]) -> None:
        self._node.set_name(value)

    def get_position(self) -> VEC_TYPE:
        return self._node.get_position()

    def set_position(self, value: VEC_TYPE) -> None:
        self._node.set_position(value)

    def add_scene_children(self, *children: 'AbstractSceneNode', **kwargs) -> None:
        self._node.add_scene_children(*children, **kwargs)

    def remove_scene_children(self, *children: 'AbstractSceneNode') -> Set["AbstractSceneNode"]:
        return self._node.remove_scene_children(*children)

    def get_scene_children(self) -> FrozenSet["AbstractSceneNode"]:
        return self._node.get_scene_children()

    def set_parent(self, parent: Optional['AbstractSceneNode']) -> None:
        self._node.set_parent(parent)

    def get_parent(self) -> Optional['AbstractSceneNode']:
        return self._node.get_parent()

    def get_global_position(self, **kwargs) -> VEC_TYPE:
        return self._node.get_global_position(**kwargs)

    def get_root(self) -> 'AbstractSceneNode':
        return self._node.get_root()
