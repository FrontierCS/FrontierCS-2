"""Import candidate modules from compiled public source after kernel sealing."""

import importlib.abc
import importlib.util
import sys


class CandidateModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, sources):
        self.modules = {}
        for name, record in sources.items():
            if name != "diffusion" and not name.startswith("diffusion."):
                raise ValueError("module outside candidate namespace")
            if not all(part.isidentifier() for part in name.split(".")):
                raise ValueError("invalid candidate module name")
            self.modules[name] = (
                bool(record["package"]),
                compile(record["source"], "<candidate:" + name + ">", "exec"),
            )

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            return importlib.util.spec_from_loader(
                fullname, self, is_package=self.modules[fullname][0])
        if fullname == "diffusion" or fullname.startswith("diffusion."):
            raise ModuleNotFoundError("candidate module not in sealed source closure")
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        package, code = self.modules[module.__name__]
        if package:
            module.__path__ = []
        exec(code, module.__dict__)

    def install(self):
        # Do not leave preloaded trusted/reference implementations discoverable.
        for name in list(sys.modules):
            if name == "diffusion" or name.startswith("diffusion."):
                del sys.modules[name]
        sys.meta_path.insert(0, self)
