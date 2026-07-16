import subprocess
import sys

def check_and_install_packages(packages):
    """
    Checks if the specified packages are installed, and if not, prompts the user
    to install them.

    Parameters:
    - packages: A list of dictionaries, each containing:
        - 'import_name': The name used in the import statement.
        - 'install_name': (Optional) The name used in the pip install command.
                          Defaults to 'import_name' if not provided.
        - 'version': (Optional) Version constraint for the package.
    """
    for package in packages:
        import_name = package['import_name']
        install_name = package.get('install_name', import_name)
        version = package.get('version', '')

        try:
            __import__(import_name)
        except ImportError:
            user_input = input(
                f"This program requires the '{import_name}' library, which is not installed.\n"
                f"Do you want to install it now? (y/n): "
            )
            if user_input.strip().lower() == 'y':
                try:
                    # Build the pip install command
                    install_command = [sys.executable, "-m", "pip", "install"]
                    if version:
                        install_command.append(f"{install_name}{version}")
                    else:
                        install_command.append(install_name)

                    subprocess.check_call(install_command)
                    __import__(import_name)
                    print(f"Successfully installed '{install_name}'.")
                except Exception as e:
                    print(f"An error occurred while installing '{install_name}': {e}")
                    sys.exit(1)
            else:
                print(f"The program requires the '{import_name}' library to run. Exiting...")
                sys.exit(1)
