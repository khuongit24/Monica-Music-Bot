"""
Custom exceptions for Monica Bot
"""

class MonicaBotError(Exception):
    """Base exception for Monica Bot."""
    pass

class ConfigurationError(MonicaBotError):
    """Raised when there's a configuration error."""
    pass

class AudioSourceError(MonicaBotError):
    """Raised when there's an audio source error."""
    pass

class ResolveError(MonicaBotError):
    """Raised when track resolution fails."""
    pass

class PlayerError(MonicaBotError):
    """Raised when there's a player error."""
    pass

class CacheError(MonicaBotError):
    """Raised when there's a cache operation error."""
    pass

class NetworkError(MonicaBotError):
    """Raised when there's a network-related error."""
    pass

class ValidationError(MonicaBotError):
    """Raised when input validation fails."""
    pass
