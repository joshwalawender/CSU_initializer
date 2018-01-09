
## Plugin Template for making a Ginga scientific viewer plugin

This is a template for creating a plugin to be used in the [Ginga
viewer](https://github.com/ejeschke/ginga).

For full instructions on writing a plugin, please [see the Ginga
documentation](http://ginga.readthedocs.io/en/latest/manual/plugins.html)
on plugins.


### Quick and Dirty Instructions for the impatient

You will want to modify the following files:

1. setup.py

   This file controls how the plugin is installed and what packages it
   needs.  Nominally, you will want to change the name, title, url, etc.

2. Copy and modify one of the two files in the "plugins" directory.

   If you are making a global type plugin (the most general) you would
   want to start with "MyGlobalPlugin.py".  If a local plugin (see the
   link above for a description of the difference) use "MyLocalPlugin.py"

3. plugins/__init__.py

   This has the functions that are actually called to fetch the information
   needed to register the plugin.

You can register one or more plugins all from the same template.

Once you think you have the plugin created correctly.  Install it using
the usual python "setup.py" installation method.  Then run the program
"util/test_registration.py" to see whether your plugins will be seen by
Ginga.




