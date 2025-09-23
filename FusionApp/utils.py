# Shared utility functions

import logging
from typing import Optional
from config_params import CFGS


def setup_logger(name, level=CFGS.LOG_LEVEL):
    """Set up a logger with consistent formatting across all modules

    Args:
        name: Name of the logger (typically the module or class name)
        level: Logging level (from CFGS.LOG_LEVEL by default)

    Returns:
        logger: Configured logger instance
    """

    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"%(asctime)s - {name} - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def disable_shm_resource_tracker(logger: Optional[logging.Logger] = None) -> None:
    """
    Disable multiprocessing.resource_tracker tracking for shared_memory objects.
    This avoids shutdown KeyErrors and misleading leak warnings when SHM is
    intentionally managed by a dedicated owner process (the engine).

    Caller must ensure explicit close()/unlink() by the owner.
    """
    try:
        # type: ignore[attr-defined]
        from multiprocessing import resource_tracker

        # 1) Remove cleanup function so the tracker won't attempt to clean SHM at exit.
        cleanup_funcs = getattr(resource_tracker, "_CLEANUP_FUNCS", None)
        if isinstance(cleanup_funcs, dict):
            cleanup_funcs.pop("shared_memory", None)

        # 2) Monkeypatch register/unregister for shared_memory to no-op, so no messages are sent.
        if not hasattr(resource_tracker, "_orig_register"):
            # type: ignore[attr-defined]
            resource_tracker._orig_register = resource_tracker.register
        if not hasattr(resource_tracker, "_orig_unregister"):
            # type: ignore[attr-defined]
            resource_tracker._orig_unregister = resource_tracker.unregister

        def _rt_register(name: str, rtype: str):  # type: ignore[no-redef]
            if rtype == "shared_memory":
                return
            # type: ignore[attr-defined]
            return resource_tracker._orig_register(name, rtype)

        def _rt_unregister(name: str, rtype: str):  # type: ignore[no-redef]
            if rtype == "shared_memory":
                return
            # type: ignore[attr-defined]
            return resource_tracker._orig_unregister(name, rtype)

        resource_tracker.register = _rt_register  # type: ignore[assignment]
        # type: ignore[assignment]
        resource_tracker.unregister = _rt_unregister

        if logger:
            logger.debug(
                "Patched resource_tracker to ignore shared_memory (engine will unlink explicitly)"
            )
    except Exception as e:
        if logger:
            logger.debug(
                "resource_tracker disable not applied (module missing or error): %s", e
            )
