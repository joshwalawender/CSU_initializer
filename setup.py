from setuptools import setup, find_packages

# You can have one or more plugins.  Just list them all here.
# For each one, add a setup function in plugins/__init__.py
#
entry_points = """
[ginga.rv.plugins]
myglobalplugin=plugins:setup_myglobalplugin
mylocalplugin=plugins:setup_mylocalplugin
"""

setup(
    name = 'MyGingaPlugins',
    version = "0.1.dev",
    description = "Plugin examples for the Ginga reference viewer",
    author = "Tycho Brahe",
    license = "BSD",
    # change this to your URL
    url = "http://ejeschke.github.com/ginga-plugin-template",
    install_requires = ["ginga>=2.6.1"],
    packages = find_packages(),
    include_package_data = True,
    package_data = {},
    entry_points = entry_points,
)
