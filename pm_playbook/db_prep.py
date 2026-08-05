"""
Initialize the PM Playbook PostgreSQL schema.

Run from the repository root with:

    uv run python -m pm_playbook.db_prep
"""

from pm_playbook.db import create_tables


def main() -> None:
    """Create the required database tables."""
    create_tables()
    print("Database tables created successfully.")


if __name__ == "__main__":
    main()
