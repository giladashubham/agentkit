__all__ = ["GoogleProvider"]


def __getattr__(name: str):
    if name == "GoogleProvider":
        from .provider import GoogleProvider

        return GoogleProvider
    raise AttributeError(name)
