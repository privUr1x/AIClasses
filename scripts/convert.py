def convert_toml_to_pip(version):
    """
    Converts a version in TOML format to PIP format.
    
    :param version_toml: Version in TOML format (e.g., ^1.21.0, >=1.2.0, <2.0.0)
    :return: Version in PIP format (e.g., >=1.21.0, <2.0.0)
    """
    # Handling patterns

    if version == "":
        return ""

    if version.startswith('^'):
        version_base = version[1:]
        major_version = int(version_base.split('.')[0])
        return f">={version_base},<{major_version + 1}.0.0"
    
    if version.startswith('~>'):
        version_base = version[2:]
        major_version = int(version_base.split('.')[0])
        return f">={version_base},<{major_version + 1}.0.0"
    
    if version.startswith('>='):
        return version
    
    if version.startswith('<'):
        return version
    
    if version.startswith('=='):
        return version
    
    if ',' in version:
        # If it's an explicit range
        return version
    
    if version == '*':
        return ''  # For any version, assuming no limit is set
    
    return f"=={version}"  # Default to exact version

if __name__ == "__main__":
    formated_version: str = convert_toml_to_pip(input("Version: "))
    print(formated_version)
