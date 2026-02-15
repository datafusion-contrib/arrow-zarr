from datafusion import SessionContext
from ._internal import ZarrTableProvider, IcechunkTableProvider


class ZarrSessionContext(SessionContext):
    """SessionContext with convenience methods for Zarr and Icechunk tables."""

    def register_zarr(self, name: str, path: str) -> None:
        """Register a Zarr store as a table.

        Args:
            name: Table name to register
            path: Path to the Zarr store (local path or s3:// URL)
        """
        self.register_table(name, ZarrTableProvider(path))

    def register_icechunk(self, name: str, path: str) -> None:
        """Register an Icechunk repository as a table.

        Args:
            name: Table name to register
            path: Path to the Icechunk repository (local path or s3:// URL)
        """
        self.register_table(name, IcechunkTableProvider(path))


__all__ = ["ZarrSessionContext", "ZarrTableProvider", "IcechunkTableProvider"]
