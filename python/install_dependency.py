import argparse
import sys

def is_venv():
  return (hasattr(sys, 'real_prefix') or
    (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def pip_install(*args):
  import subprocess  # nosec - disable B404:import-subprocess check

  cli_args = []
  for arg in args:
    cli_args.extend(str(arg).split(" "))
  subprocess.run([sys.executable, "-m", "pip", "install", *cli_args], check=True)

def pip_uninstall(*args):
  import subprocess  # nosec - disable B404:import-subprocess check

  cli_args = []
  for arg in args:
    cli_args.extend(str(arg).split(" "))
  subprocess.run([sys.executable, "-m", "pip", "uninstall", *cli_args], check=True)

def install_dep():
  print("Installing dependencies...")
  print(f"Python version: {sys.version}")

  # Check Python version compatibility
  if sys.version_info >= (3, 13):
    print(f"WARNING: Python {sys.version_info.major}.{sys.version_info.minor} detected.")
    print("onnxruntime may not support this version yet.")
    print("Recommended: Use Python 3.11 or 3.12")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
      sys.exit(1)

  packages = [
    "python-dotenv",
    "numpy",
    "opencv-python",
    "pyyaml",
    "requests",
  ]

  # Install onnxruntime — prefer GPU build if --gpu flag was passed
  ort_package = "onnxruntime-gpu" if globals().get("_args") and globals()["_args"].gpu else "onnxruntime"

  for pkg in packages:
    try:
      print(f"  Installing {pkg} …")
      pip_install(pkg)
    except Exception as e:
      print(f"\nInstallation failed for '{pkg}': {e}")
      print("\nTroubleshooting:")
      print("1. Use Python 3.11 or 3.12")
      print("2. Upgrade pip: python -m pip install --upgrade pip")
      print("3. Try installing packages individually to identify which one fails")
      sys.exit(1)

  try:
    print(f"  Installing {ort_package} …")
    pip_install(ort_package)
  except Exception as e:
    if ort_package == "onnxruntime-gpu":
      print(f"  GPU build failed ({e}), falling back to CPU onnxruntime …")
      try:
        pip_install("onnxruntime")
      except Exception as e2:
        print(f"\nInstallation failed for 'onnxruntime': {e2}")
        sys.exit(1)
    else:
      print(f"\nInstallation failed for '{ort_package}': {e}")
      sys.exit(1)

  print("\nAll dependencies installed successfully.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Install project dependencies")
  parser.add_argument(
    "--gpu", action="store_true",
    help="Install onnxruntime-gpu instead of the CPU-only onnxruntime package.",
  )
  _args = parser.parse_args()

  if is_venv():
    install_dep()
  else:
    print("Not running inside a virtual environment")
