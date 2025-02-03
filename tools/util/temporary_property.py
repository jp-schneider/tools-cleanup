from tools.logger.logging import tools_logger as logger, get_messaged
from tools.util.format import get_frame_summary
from traceback import FrameSummary, extract_stack
if not get_messaged("temporary_property"):
    raise ImportError(
        "The import of TemporaryProperty from tools.util.temporary_property is deprecated. Please use tools.context.temporary_property instead.")
    logger.warning(
        "The import of TemporaryProperty from tools.util.temporary_property is deprecated. Please use tools.context.temporary_property instead.")
from tools.context.temporary_property import TemporaryProperty
