def enhance_prompt(base_prompt: str) -> str:
    """
    Appends standard diffusion modifiers to a prompt for better quality output.

    Args:
        base_prompt (str): The user-provided prompt.

    Returns:
        str: Enhanced prompt with quality modifiers.
    """
    modifiers = [
        "high resolution",
        "HD",
        "ultra-detailed",
        "sharp focus",
    ]

    # Avoid duplicating modifiers
    enhanced = base_prompt.strip()
    for modifier in modifiers:
        if modifier.lower() not in enhanced.lower():
            enhanced += f", {modifier}"

    return enhanced
