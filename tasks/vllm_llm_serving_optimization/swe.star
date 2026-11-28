"""The SWE-bench instances, in the order of `judge/instance_manifest.json`: the five of the train
phase, then those that only the final evaluation adds to them. Each has its slot in the parent's
pod, which names its containers `swe-<slot>-candidate` and `swe-<slot>-check`
(`judge/sidecar_catalog.py`), its SWE-bench image by digest, and the build arguments of its source
preparation, which `swe/Dockerfile` explains. Slots keep the source's numbering, from which 018 and
039 were removed."""

TRAIN_SWE_INSTANCES = [
    {
        "id": "scikit-learn__scikit-learn-11281",
        "slot": "000",
        "base": "docker.io/swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-11281@sha256:bc89ff7e6838610f8b84668c9284f50d6e336abfc27fa6d79805148a06245975",
    },
    {
        "id": "django__django-13925",
        "slot": "001",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-13925@sha256:dfcffa35bc68e3b014c6e4c041210721cfc17179fd372e4f7385c6e6384d5db6",
    },
    {
        "id": "pytest-dev__pytest-5495",
        "slot": "002",
        "base": "docker.io/swebench/sweb.eval.x86_64.pytest-dev_1776_pytest-5495@sha256:2a435aad4e26c0ed4f2f846615db06696bf153975dcef2d4f7b9eb83f1ba13b3",
    },
    {
        "id": "sympy__sympy-12236",
        "slot": "003",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-12236@sha256:be9f26a78e7bcbabc304cd0edd1b10e5c9d3bc0f647cb17b7aa99763d8d02330",
    },
    {
        "id": "scikit-learn__scikit-learn-10949",
        "slot": "004",
        "base": "docker.io/swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-10949@sha256:a7fda995ed0b53871a0166f4cbfe78109ece96f62f52f948b84677a6599dda25",
    },
]

FINAL_ONLY_SWE_INSTANCES = [
    {
        "id": "django__django-11179",
        "slot": "005",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-11179@sha256:f36d81831615d46d1eea34feb6c49a2815f637a28bca14105db346bf90a8bd0d",
    },
    {
        "id": "mwaskom__seaborn-3010",
        "slot": "006",
        "base": "docker.io/swebench/sweb.eval.x86_64.mwaskom_1776_seaborn-3010@sha256:9f0f58e6cdffc87de2276e55d2f878155d52bfe403d7621d52f3d39c12b3deda",
    },
    {
        "id": "sphinx-doc__sphinx-8595",
        "slot": "007",
        "base": "docker.io/swebench/sweb.eval.x86_64.sphinx-doc_1776_sphinx-8595@sha256:678b5c38c6aecdbb6b743e72524bb7ef7e16889b7d022ce2645d0fcf1071eac0",
        "source": {
            "TEST_SANDBOX_SOURCE": "restored-ra",
            "TEST_SANDBOX_RESTORE_COMMIT": "b19bce971e82f2497d67fdacdeca8db08ae0ba56",
            "TEST_SANDBOX_PYTEST_ADDOPTS": "-rA",
        },
    },
    {
        "id": "django__django-16041",
        "slot": "008",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-16041@sha256:0138239ad8eb5af6773432f42e1bacaa8afed4c6398d21473df940c247cdce43",
    },
    {
        "id": "django__django-15819",
        "slot": "009",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-15819@sha256:7a2f18e67120eda350afffe769379a80fe8d2b293b7d23d8ac648bb5721a3140",
    },
    {
        "id": "sympy__sympy-21612",
        "slot": "010",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-21612@sha256:689024e8a43de562fa754cc1766e73dc4f4abb39b5e11dd01ce177addec4eb47",
    },
    {
        "id": "django__django-15790",
        "slot": "011",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-15790@sha256:99bc93ddba0d1bf6a585ade93e85ec3ae1ed92e8376fffed0eb26e6f12108ce6",
    },
    {
        "id": "django__django-16139",
        "slot": "012",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-16139@sha256:3dc112cdde1abd0b6f4ed9f4c7636df2bd194c5d023cda23d34bec90005c9bdd",
    },
    {
        "id": "django__django-13768",
        "slot": "013",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-13768@sha256:2000bd17877656512b64f7a91fa1bd24321dbd48f6bf4d44b406a519b0a12b61",
    },
    {
        "id": "sympy__sympy-15308",
        "slot": "014",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-15308@sha256:13edc0fe53d44216af865ac94a327dc899f01555f440ef4abb9951b986c283a1",
    },
    {
        "id": "django__django-16229",
        "slot": "015",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-16229@sha256:f9eb4ca84aff653b3b498f0a9ea82c7ed9535a5be22435ed415d8a7f652e3a9c",
    },
    {
        "id": "django__django-13447",
        "slot": "016",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-13447@sha256:d46931e1778c2325692eea457317da9a6eb73b086b53553a2f10d6427970d586",
    },
    {
        "id": "sympy__sympy-14817",
        "slot": "017",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-14817@sha256:8f327005ca8ebe19032c66752c848201ab4d900953b2514e50d49f2d02ae6886",
    },
    {
        "id": "sympy__sympy-21847",
        "slot": "019",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-21847@sha256:dc63eae1d6089ed44748eca7a136b877beaf316ca0ed31d0c563909ff1e094c8",
    },
    {
        "id": "sympy__sympy-18189",
        "slot": "020",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-18189@sha256:5d4dcec6309087223683586740b4fd96e27c0ef1678bccc872a0dd4eba1743b9",
    },
    {
        "id": "django__django-11099",
        "slot": "021",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-11099@sha256:364ab4524539dda8fc9c18c7b4761c76113eba606b79b845563d5052dec163c4",
    },
    {
        "id": "sympy__sympy-15609",
        "slot": "022",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-15609@sha256:7742822e67417b1779643561b8c35b808dd35ef9b6bd4268457911add99c4be3",
    },
    {
        "id": "sympy__sympy-13773",
        "slot": "023",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-13773@sha256:a65489962e6f42ce0970402b26ac5b9f36ba8374bba9c53614f094f4b6d62a78",
    },
    {
        "id": "sympy__sympy-15678",
        "slot": "024",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-15678@sha256:21b08cae68906949287847364976ffb90659dc9bc6cc2707ad1290d34cd9622a",
    },
    {
        "id": "astropy__astropy-14365",
        "slot": "025",
        "base": "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-14365@sha256:52047c9299800168ed38a1b98b518a62c32f407f8269362c9f381d93c4e69dfe",
        "source": {
            "TEST_SANDBOX_SOURCE": "restored",
            "TEST_SANDBOX_RESTORE_COMMIT": "7269fa3e33e8d02485a647da91a5a2a60a06af61",
        },
    },
    {
        "id": "astropy__astropy-14182",
        "slot": "026",
        "base": "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-14182@sha256:4a1da544b0a2d7bb43266093db899f77c60700f76b84497e6c66138cb30d8663",
        "source": {
            "TEST_SANDBOX_SOURCE": "restored",
            "TEST_SANDBOX_RESTORE_COMMIT": "a5917978be39d13cd90b517e1de4e7a539ffaa48",
        },
    },
    {
        "id": "scikit-learn__scikit-learn-25638",
        "slot": "027",
        "base": "docker.io/swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-25638@sha256:79f1999c287bebe3d7a593f398ca63245332a1e09bac45de82d26670d4974566",
    },
    {
        "id": "django__django-13710",
        "slot": "028",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-13710@sha256:ffe2e2263ad63e6f0b42adffac063682f953cf4500882b3c4d9429f0d17845a8",
    },
    {
        "id": "django__django-14752",
        "slot": "029",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-14752@sha256:df8caa4518b9d1295e55052e61c939bb583e33dbd57db7d44693bc27077c426d",
    },
    {
        "id": "django__django-12747",
        "slot": "030",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-12747@sha256:db88078ddec0bdee9e9d115acf0c43a937533f3d4e876c5345574169928ecd52",
    },
    {
        "id": "django__django-12856",
        "slot": "031",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-12856@sha256:d4ba4633506e0b2fae3c0880f07a27300d31a2652524af465c0622db27d6c15c",
    },
    {
        "id": "django__django-16255",
        "slot": "032",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-16255@sha256:6be06e9f3f8f648539f2aebf68a2630c49df6b6632a1cf889d61b0f6677fa3a2",
    },
    {
        "id": "django__django-11620",
        "slot": "033",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-11620@sha256:6d7f9546fee337c0251c010e2ce72f9bd2ce2c947e36b80ea708ad379c222a83",
    },
    {
        "id": "sympy__sympy-23117",
        "slot": "034",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-23117@sha256:fcce4c9f68eac8ca121b28de274361bf927a84ebdfe6ebdeb54601b6af433eb4",
    },
    {
        "id": "pylint-dev__pylint-7080",
        "slot": "035",
        "base": "docker.io/swebench/sweb.eval.x86_64.pylint-dev_1776_pylint-7080@sha256:d81b3855816c07747fbfd694352ce24fb8152cec70d136b292de55ff6afe2db6",
    },
    {
        "id": "pytest-dev__pytest-5227",
        "slot": "036",
        "base": "docker.io/swebench/sweb.eval.x86_64.pytest-dev_1776_pytest-5227@sha256:39a8b42bd65fcbf1870cdd7fc58b0c65d56546e69f06ab32bcdb693751653c34",
    },
    {
        "id": "django__django-11910",
        "slot": "037",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-11910@sha256:7d87ee77e7c2f70b2db8b9ec8814536371cdf18c7a85625003664c57e714a211",
    },
    {
        "id": "django__django-14787",
        "slot": "038",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-14787@sha256:f029159b39942836a8c09738bda245bfea9d197e1b94f23e3eeaf576e3008600",
    },
    {
        "id": "sphinx-doc__sphinx-7738",
        "slot": "040",
        "base": "docker.io/swebench/sweb.eval.x86_64.sphinx-doc_1776_sphinx-7738@sha256:0df692a53066cd84630a1bab754f4b35c10d2c994049c90fbba54aae791e7cf3",
        "source": {
            "TEST_SANDBOX_SOURCE": "restored-ra",
            "TEST_SANDBOX_RESTORE_COMMIT": "c087d717f6ed183dd422359bf91210dc59689d63",
            "TEST_SANDBOX_PYTEST_ADDOPTS": "-rA",
        },
    },
    {
        "id": "sympy__sympy-17139",
        "slot": "041",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-17139@sha256:bdd9cb046564ab53dc89940213b8dd7cce35c87a62b8ccb85cbe8b337cc26a39",
    },
    {
        "id": "sympy__sympy-11897",
        "slot": "042",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-11897@sha256:38a90fcda3190d5ee401ba244a38316438645a041feaf9cfca1ef93f980a9e40",
    },
    {
        "id": "django__django-14608",
        "slot": "043",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-14608@sha256:115014157acb675b51ffbdf53e9b546a207978c630ab3a8c686d2aedef62efab",
    },
    {
        "id": "sympy__sympy-19487",
        "slot": "044",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-19487@sha256:3289b9f407aee8220ef6156268ea4c909eb9f50b0c69238688cf28ce74fa585c",
    },
    {
        "id": "astropy__astropy-7746",
        "slot": "045",
        "base": "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-7746@sha256:eb09bc366b912d0d6092512376469f1451cdf692066110f30c5ff8fc38ab5c31",
    },
    {
        "id": "django__django-16910",
        "slot": "046",
        "base": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-16910@sha256:20817b3f2d8ee1f0c5c4fb85fe26954937341f78d9ab067ce8b0722126d624fc",
    },
    {
        "id": "psf__requests-3362",
        "slot": "047",
        "base": "docker.io/swebench/sweb.eval.x86_64.psf_1776_requests-3362@sha256:d1c8c15eeb5e12b7eb3815307d1131d68c72d53419fa2f8b89bc46bc0009a61b",
    },
    {
        "id": "sympy__sympy-13043",
        "slot": "048",
        "base": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-13043@sha256:cbfb8d7777b7a65fb868e2aa367b6bdd40f5021456dabf46acad549f4837624b",
    },
    {
        "id": "astropy__astropy-6938",
        "slot": "049",
        "base": "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-6938@sha256:e71ac0096dd19e742e2e70d088e970ee10471347847409c89cd40dbf32f4715e",
    },
]
