import pathlib
import sys

# Stage scripts live in scripts/ and are imported by the smoke tests.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
