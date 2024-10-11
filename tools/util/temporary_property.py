from tools.logger.logging import tools_logger as logger, get_messaged

if not get_messaged("temporary_property"):
    logger.warning(
        "The import of TemporaryProperty from tools.util.temporary_property is deprecated. Please use tools.context.temporary_property instead.")
from tools.context.temporary_property import TemporaryProperty
