"""ComfyUI nodes for local GGUF vision-language models, Wildcards, and Krea2 Detailers."""

if not __package__:
    import sys
    import types
    from pathlib import Path

    _pkg_name = "_ziggart_root_pkg"
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(__file__).resolve().parent)]
        sys.modules[_pkg_name] = _pkg
    __package__ = _pkg_name
    if __spec__ is not None:
        __spec__.name = _pkg_name
        __spec__.submodule_search_locations = [str(Path(__file__).resolve().parent)]

from .nodes import (
    NODE_CLASS_MAPPINGS as BASE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as BASE_DISPLAY_MAPPINGS,
)
from .detailer_nodes import (
    NODE_CLASS_MAPPINGS as DETAILER_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as DETAILER_DISPLAY_MAPPINGS,
)
from .addon_nodes import (
    NODE_CLASS_MAPPINGS as ADDON_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ADDON_DISPLAY_MAPPINGS,
)
from .wildcard_nodes import (
    NODE_CLASS_MAPPINGS as WILDCARD_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as WILDCARD_DISPLAY_MAPPINGS,
)
from .three_d_nodes import (
    NODE_CLASS_MAPPINGS as THREED_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as THREED_DISPLAY_MAPPINGS,
)
from .attention_nodes import (
    NODE_CLASS_MAPPINGS as ATTENTION_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ATTENTION_DISPLAY_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {
    **BASE_CLASS_MAPPINGS,
    **DETAILER_CLASS_MAPPINGS,
    **ADDON_CLASS_MAPPINGS,
    **WILDCARD_CLASS_MAPPINGS,
    **THREED_CLASS_MAPPINGS,
    **ATTENTION_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **BASE_DISPLAY_MAPPINGS,
    **DETAILER_DISPLAY_MAPPINGS,
    **ADDON_DISPLAY_MAPPINGS,
    **WILDCARD_DISPLAY_MAPPINGS,
    **THREED_DISPLAY_MAPPINGS,
    **ATTENTION_DISPLAY_MAPPINGS,
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
