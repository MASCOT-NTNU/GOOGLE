"""
Checks if the folder exists or not.
"""

import pathlib


def checkfolder(folder):
    path = pathlib.Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    # print(folder + "is created")
