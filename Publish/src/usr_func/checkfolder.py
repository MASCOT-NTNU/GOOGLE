"""
Check if the folder exists, if not, create it.
"""

import pathlib


def checkfolder(folder) -> None:
    """
    Args:
        folder: str, the folder to be checked.

    Returns:
        None

    """
    path = pathlib.Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    # print(folder + "is created")
