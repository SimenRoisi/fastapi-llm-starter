import bcrypt


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key using bcrypt.
    
    Args:
        api_key: The plain text API key to hash
        
    Returns:
        The bcrypt hash as a string (includes salt and cost factor)
        
    Example:
        >>> hash_api_key("my-secret-key")
        "$2b$12$xyz123..."
    """
    # Convert string to bytes (bcrypt requires bytes)
    api_key_bytes = api_key.encode('utf-8')
    
    # Generate a salt with cost factor 12 (2^12 = 4,096 rounds)
    salt = bcrypt.gensalt(rounds=12)
    
    # Hash the API key with the salt
    hashed = bcrypt.hashpw(api_key_bytes, salt)
    
    # Convert back to string for database storage
    return hashed.decode('utf-8')


def verify_api_key(api_key: str, hashed_api_key: str) -> bool:
    """
    Verify an API key against its bcrypt hash.
    
    Args:
        api_key: The plain text API key to verify
        hashed_api_key: The stored bcrypt hash
        
    Returns:
        True if the API key matches the hash, False otherwise
        
    Example:
        >>> verify_api_key("my-secret-key", "$2b$12$xyz123...")
        True
    """
    try:
        # Convert both to bytes
        api_key_bytes = api_key.encode('utf-8')
        hashed_bytes = hashed_api_key.encode('utf-8')
        
        # bcrypt.checkpw handles the comparison
        return bcrypt.checkpw(api_key_bytes, hashed_bytes)
    except (ValueError, TypeError):
        # Invalid hash format or other errors
        return False
