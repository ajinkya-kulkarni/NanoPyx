#!/bin/bash -c python3

import os
import sys
from inspect import isfunction


def find_files(root_dir, extension):
    target_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                target_files.append(os.path.join(root, file))

        # auto remove empty directories
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if os.listdir(dir_path) == []:
                print("Removing empty directory: ", dir_path)
                os.rmdir(dir_path)
    return target_files


def update_gitignore():
    gitignore_lines = open(".gitignore", "r").read().splitlines()
    ignores = [
        ".coverage*",
        "tests",
        "tests_plots",
    ]
    for ignore in ignores:
        if ignore not in gitignore_lines:
            gitignore_lines.append(ignore)

    with open(".gitignore", "w") as f:
        f.write("\n".join(gitignore_lines))


def main():

    clean_files = " ".join(
        find_files("src", ".so")
        + find_files("src", ".pyc")
        + find_files("src", ".c")
        + find_files("src", ".html")
        + find_files("src", ".profile")
        + find_files("notebooks", ".profile")
    )

    notebook_files = " ".join(find_files("notebooks", ".ipynb"))
    options = {
        "Build nanopyx extensions": "python3 setup.py build_ext --inplace",
        "Auto-generate pxd files with pyx2pxd": f"pyx2pxd src",
        "Clean files": f"rm {clean_files}"
        if len(clean_files) > 0
        else "echo 'No files to clean'",
        "Clear notebook output": f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {notebook_files}",
        "Update .gitignore": update_gitignore,
        "Run pdoc": "python -m pdoc src/nanopyx -o docs",
        "Install nanopyx in developer mode": "pip3 install -e .[developer, test]",
        "Build nanopyx binary distribution": "python3 setup.py bdist_wheel",
        "Build nanopyx source distribution": "python3 setup.py sdist",
        "Install coding tools": "pip install cython-lint",
        "Run cython-lint on pyx files": f"cython-lint {', '.join(find_files('src', '.pyx'))}",
    }

    print(
        """
                         ,.
                        (_|,.
                       ,' /, )_______   _
                    __j o``-'        `.'-)'
                   ('')     NanoPyx     '
                    `-j                |
                      `-._(           /
         Oink! Oink!     |_\  |--^.  /
        |--- nm ---|    /_]'|_| /_)_/
                            /_]'  /_]'
    """
    )

    if len(sys.argv) > 1:
        selection = int(sys.argv[1]) - 1

    else:
        # print the options
        print("What do you want to do:")
        for i, option in enumerate(options.keys()):
            cmd = options[option]
            if type(cmd) == str:
                print(
                    f"{i+1}) {option}: [CMD]> {cmd if len(cmd)< 100 else cmd[:100]+'...'}"
                )
            elif isfunction(cmd):
                print(f"{i+1}) {option}: [FUNCTION]> {repr(cmd)}")

        # get the user's selection
        selection = int(input("Enter your selection: ")) - 1

    # print the selected option
    cmd = list(options.values())[selection]
    print(f"- Running command: {repr(cmd)}")
    if type(cmd) == str:
        os.system(cmd)
    elif isfunction(cmd):
        cmd()

if __name__ == "__main__":
    main()
