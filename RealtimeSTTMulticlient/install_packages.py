import subprocess
import sys
import importlib

def check_and_install_packages(packages):
    """
    Checks if the specified packages are installed, and if not, prompts the user
    to install them.

    Parameters:
    - packages: A list of dictionaries, each containing:
        - 'module_name': The module or package name to import.
        - 'attribute': (Optional) The attribute or class to check within the module.
        - 'install_name': The name used in the pip install command.
        - 'version': (Optional) Version constraint for the package.
    """
    for package in packages:
        module_name = package['module_name']
        attribute = package.get('attribute')
        install_name = package.get('install_name', module_name)
        version = package.get('version', '')

        try:
            # Attempt to import the module
            module = importlib.import_module(module_name)
            # If an attribute is specified, check if it exists
            if attribute:
                getattr(module, attribute)
        except (ImportError, AttributeError):
            user_input = input(
                f"This program requires '{module_name}'"
                f"{'' if not attribute else ' with attribute ' + attribute}, which is not installed or missing.\n"
                f"Do you want to install '{install_name}' now? (y/n): "
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
                    # Try to import again after installation
                    module = importlib.import_module(module_name)
                    if attribute:
                        getattr(module, attribute)
                    print(f"Successfully installed '{install_name}'.")
                except Exception as e:
                    print(f"An error occurred while installing '{install_name}': {e}")
                    sys.exit(1)
            else:
                print(f"The program requires '{install_name}' to run. Exiting...")
                sys.exit(1)
