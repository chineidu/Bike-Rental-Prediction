def get_vibrant_color(correlation: float) -> str:
    """Generate vibrant colors based on correlation strength and direction."""
    abs_corr = abs(correlation)

    if correlation > 0:
        # Positive correlations: vibrant blues to greens
        if abs_corr > 0.7:
            return "#00FF41"  # Bright green
        if abs_corr > 0.5:
            return "#00D4AA"  # Teal
        if abs_corr > 0.3:
            return "#00B4D8"  # Cyan
        return "#0077BE"  # Blue
    # Negative correlations: vibrant reds to oranges
    if abs_corr > 0.7:
        return "#FF073A"  # Bright red
    if abs_corr > 0.5:
        return "#FF4500"  # Orange red
    if abs_corr > 0.3:
        return "#FF6B35"  # Orange
    return "#FFB000"  # Amber


def _select_valid_columns(
    actual_cols: list[str], selected_cols: list[str]
) -> list[str]:
    """Select valid columns from the actual columns based on user selection."""
    return list(set(actual_cols) & set(selected_cols))
