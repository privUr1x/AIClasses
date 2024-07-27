from toml import load 
from convert import convert_toml_to_pip

toml_path: str = "/".join(__file__.split("/")[:-1].__add__(["..", "pyproject.toml"]))
req_path: str = "/".join(__file__.split("/")[:-1].__add__(["..", "requirements.txt"]))

def main(tpath: str, rpath: str) -> None:

    with open(tpath, "r") as f:
        toml: dict = load(f)

    data: dict = toml["tool"]["poetry"]["dependencies"]
    data.pop("python")

    content: list[str] = [k + convert_toml_to_pip(v) for k, v in data.items()]

    with open(rpath, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
        print("Done!")

if __name__ == "__main__":
    main(toml_path, req_path)
