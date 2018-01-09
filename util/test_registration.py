"""
This program allows you to test whether your plugin will register
itself correctly.
"""
from pkg_resources import iter_entry_points

groups = ['ginga.rv.plugins']

available_methods = []

for group in groups:
    for entry_point in iter_entry_points(group=group, name=None):
        available_methods.append(entry_point.load())

for method in available_methods:
    spec = method()
    print(spec)
