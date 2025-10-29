"""
Repository structure visualization utilities for displaying directory trees.
"""
import os
import pathlib
from rich import print
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree
from io import StringIO
from rich.console import Console


'''
Warning!!!!
If you see the following error in windows
[WinError 3] The system cannot find the path specified:
'playground\\java\\jitsi__jitsi-319309e\\repo\\resources\\install\\src\\mac\\dist\\Jitsi.app\\Contents\\Frameworks\\Spar
kle.framework\\Versions\\A\\Resources\\finish_installation.app\\Contents\\Resources\\pt_BR.lproj'
you need to enable long path in windows setting
'''

def walk_directory(directory: pathlib.Path, tree: Tree, max_depth: int, current_depth: int = 0) -> None:
    """
    Recursively build a Tree with directory contents, stopping at max_depth.
    
    Args:
        directory (pathlib.Path): Directory to traverse
        tree (Tree): Rich Tree object to populate
        max_depth (int): Maximum depth to traverse (-1 for unlimited)
        current_depth (int): Current traversal depth
    """
    if max_depth != -1 and current_depth >= max_depth:
        return

    ignore_dirs = [".git", ".svn", "__pycache__"]
    ignore_files = [".DS_Store", ".gitignore", ".gitattributes"]
    paths = sorted(
        pathlib.Path(directory).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )
    for path in paths:
        if path.is_dir():
            if path.name in ignore_dirs:
                continue
            branch = tree.add(
                f"[bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}"
            )
            walk_directory(path, branch, max_depth, current_depth + 1)
        else:
            if path.name in ignore_files:
                continue
            text_filename = Text(path.name, "green")
            text_filename.highlight_regex(r"\..*$", "bold red")
            text_filename.stylize(f"link file://{path}")
            icon = "🐍 " if path.suffix == ".py" else "📄 "
            tree.add(Text(icon) + text_filename)

def view_repo_structure(directory: str, max_depth: int = -1) -> str:
    """
    Generate a string representation of the repository folder structure.
    
    Args:
        directory (str): Path to the directory to visualize
        max_depth (int): Maximum depth to traverse (-1 for unlimited)
        
    Returns:
        str: String representation of the directory tree structure
        
    Raises:
        ValueError: If directory is not a valid directory path
    """
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory.")
    
    tree = Tree(
        f":open_file_folder: [link file://{directory}]{directory}",
        guide_style="bold bright_blue",
    )
    walk_directory(pathlib.Path(directory), tree, max_depth=max_depth)

    console = Console(file=StringIO())
    console.print(tree)
    str_output = console.file.getvalue()
    return str_output
