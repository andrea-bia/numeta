class NativeNameRegistry:
    """Track generated native symbols that may remain loaded by the process.

    ``active_names`` holds names currently owned by live NumetaFunctions.
    ``reserved_names`` holds *all* names ever allocated — these can never be
    safely reused within the process because the old .so may still be loaded.
    """

    def __init__(self):
        self.active_names: set[str] = set()
        self.reserved_names: set[str] = set()

    def is_active(self, name: str) -> bool:
        return name in self.active_names

    def is_reserved(self, name: str) -> bool:
        return name in self.active_names or name in self.reserved_names

    def reserve(self, name: str) -> None:
        self.active_names.add(name)
        self.reserved_names.add(name)

    def reserve_many(self, names) -> None:
        self.active_names.update(names)
        self.reserved_names.update(names)

    def release_active(self, name: str) -> None:
        self.active_names.discard(name)

    def default_candidate(self, base_name: str) -> tuple[str, bool]:
        suffix = len(self.active_names)
        name = f"{base_name}_{suffix}"
        first_candidate_was_active = self.is_active(name)

        while self.is_reserved(name):
            suffix += 1
            name = f"{base_name}_{suffix}"

        return name, first_candidate_was_active


native_name_registry = NativeNameRegistry()
