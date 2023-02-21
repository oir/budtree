import os


def ensure_file(fname: str, url: str) -> str:
    if os.path.exists(fname):
        return fname

    print(f"Could not find file {fname}, attempting to download from {url}...")
    import wget

    wget.download(url, fname)
    return fname
