__all__ = [
    "DummyDataset",
    "EpisodicRLDSDataset",
    "LeRobotBatchTransform",
    "LeRobotDataset",
    "RLDSBatchTransform",
    "RLDSDataset",
]


def __getattr__(name):
    if name in {"DummyDataset", "EpisodicRLDSDataset", "RLDSBatchTransform", "RLDSDataset"}:
        from .datasets import DummyDataset, EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset

        return {
            "DummyDataset": DummyDataset,
            "EpisodicRLDSDataset": EpisodicRLDSDataset,
            "RLDSBatchTransform": RLDSBatchTransform,
            "RLDSDataset": RLDSDataset,
        }[name]

    if name in {"LeRobotBatchTransform", "LeRobotDataset"}:
        from .lerobot import LeRobotBatchTransform, LeRobotDataset

        return {
            "LeRobotBatchTransform": LeRobotBatchTransform,
            "LeRobotDataset": LeRobotDataset,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
