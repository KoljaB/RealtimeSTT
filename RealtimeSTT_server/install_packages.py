import subprocess
import sys
import importlib
import os

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}: {e}")
        return False
    return True

def check_and_install_packages(packages):
    """Check if packages are installed and install them if not."""
    for package in packages:
        try:
            # Whitelist of allowed modules for security
            ALLOWED_MODULES = {
                'numpy', 'scipy', 'pandas', 'matplotlib', 'requests', 'flask',
                'torch', 'tensorflow', 'sklearn', 'cv2', 'PIL', 'openai',
                'whisper', 'pyaudio', 'sounddevice', 'wave', 'threading',
                'queue', 'time', 'json', 'logging', 'argparse', 'pathlib',
                'collections', 'functools', 'itertools', 'typing', 'dataclasses'
            }
            
            # Validate module name against whitelist
            if package not in ALLOWED_MODULES:
                raise ImportError(f"Module '{package}' is not in the allowed modules list")
                
            importlib.import_module(package)
            print(f"{package} is already installed")
        except ImportError:
            print(f"{package} not found, installing...")
            if not install_package(package):
                print(f"Failed to install {package}")
                return False
    return True

if __name__ == "__main__":
    # Example usage
    required_packages = [
        "numpy",
        "requests",
        "flask"
    ]
    
    if check_and_install_packages(required_packages):
        print("All packages are ready!")
    else:
        print("Some packages failed to install")
        sys.exit(1)
