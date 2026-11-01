"""Public dataset error boundary shared by adapters and storage."""


class DatasetError(Exception):
    """A dataset is absent, malformed, unvalidated, or incompatible."""
