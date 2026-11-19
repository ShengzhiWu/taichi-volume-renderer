"""Compatibility entry point for the sparse smoke viewer."""

if __package__:
    from .sparse_smoke import main
else:
    from sparse_smoke import main


if __name__ == "__main__":
    main()
