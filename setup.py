from setuptools import setup, find_packages

# You can have one or more plugins.  Just list them all here.
# For each one, add a setup function in plugins/__init__.py
#
entry_points = """
[ginga.rv.plugins]
CSU_initializer=CSU_initializer_plugin:setup_CSU_initializer
"""

setup(
    name = 'CSU_initializer',
    version = "0.1.dev",
    description = "Tool for initializing the MOSFIRE CSU",
    author = "Josh Walawender",
    license = "BSD",
    # change this to your URL
    url = "https://github.com/joshwalawender/CSU_initializer",
    install_requires = ["ginga>=2.6.1"],
    packages = find_packages(),
    include_package_data = True,
    package_data = {},
    entry_points = entry_points,
)
