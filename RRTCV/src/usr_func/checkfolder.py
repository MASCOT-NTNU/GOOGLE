"""
Utility function to check if a folder exists and create it if it doesn't.
"""
import os


def checkfolder(folderpath: str) -> None:
    """
    Check if a folder exists and create it if it doesn't.
    
    Args:
        folderpath: Path to the folder to check/create
    """
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        print(f"Created folder: {folderpath}") 