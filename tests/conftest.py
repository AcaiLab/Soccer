import pathlib
import sys

import pytest

# Stage scripts live in scripts/ and are imported by the smoke tests.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

DROPBOX_VARIABLES = ("DROPBOX_TOKEN", "DROPBOX_TOKEN_FILE", "DROPBOX_SHARED_URL", "DROPBOX_LINK_PASSWORD",
                     "DROPBOX_APP_KEY", "DROPBOX_APP_SECRET", "DROPBOX_REFRESH_TOKEN", "DROPBOX_OAUTH_FILE",
                     "SOCCER_SSM_PREFIX")


@pytest.fixture
def no_dropbox_env(monkeypatch, tmp_path):
    """No Dropbox settings from the environment, and none from the default files in tmp_dbx/."""
    for name in DROPBOX_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # the default credential files are relative to the working directory
