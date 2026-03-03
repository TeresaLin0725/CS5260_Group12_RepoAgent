import adalflow as adal

from api.config import configs, get_embedder_type


def get_embedder(embedder_type: str = None, **kwargs) -> adal.Embedder:
    """Get embedder based on configuration or parameters.
    
    Args:
        embedder_type: Direct specification of embedder type ('google', 'bedrock', 'openai')
    
    Returns:
        adal.Embedder: Configured embedder instance
    """
    # Determine which embedder config to use
    if embedder_type:
        if embedder_type == 'google':
            embedder_config = configs["embedder_google"]
        elif embedder_type == 'bedrock':
            embedder_config = configs["embedder_bedrock"]
        else:  # default to openai
            embedder_config = configs["embedder"]
    else:
        # Auto-detect based on current configuration
        current_type = get_embedder_type()
        if current_type == 'bedrock':
            embedder_config = configs["embedder_bedrock"]
        elif current_type == 'google':
            embedder_config = configs["embedder_google"]
        else:
            embedder_config = configs["embedder"]

    # --- Initialize Embedder ---
    model_client_class = embedder_config["model_client"]
    if "initialize_kwargs" in embedder_config:
        model_client = model_client_class(**embedder_config["initialize_kwargs"])
    else:
        model_client = model_client_class()
    
    # Create embedder with basic parameters
    embedder_kwargs = {"model_client": model_client, "model_kwargs": embedder_config["model_kwargs"]}
    
    embedder = adal.Embedder(**embedder_kwargs)
    
    # Set batch_size as an attribute if available (not a constructor parameter)
    if "batch_size" in embedder_config:
        embedder.batch_size = embedder_config["batch_size"]
    return embedder
