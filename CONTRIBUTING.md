# Contributing code

1. Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html).
2. (Optional, but recommended) If you are using a conda version `<=23.9.0`, set the conda solver to libmamba for faster dependency solving. Starting from conda version [`23.10.0`](https://github.com/conda/conda/releases/tag/23.10.0), libmamba is the default solver.
    ```bash
    conda config --set solver libmamba
    ```
3. Clone the SustainGym repo, and enter the `sustaingym` directory.
    ```bash
    git clone https://github.com/chrisyeh96/sustaingym.git
    cd sustaingym
    ```
4. Create conda environment. Replace `XX` below with the name of the SustainGym environment you want to work on. By default, the `env_XX.yml` environment files assume that you have a NVIDIA GPU. If you do not have a NVIDIA GPU, you may need to modify the `env_XX.yml` file.
    ```bash
    conda env update --file env_XX.yml --prune
    ```
5. Make code modifications in a separate git branch
    ```bash
    git checkout -b new_feature
    ```
6. From repo root folder, run mypy type checker and fix any errors.
    ```bash
    mypy sustaingym
    ```
7. From repo root folder, run code linter and fix any linting errors.
    ```bash
    flake8 sustaingym
    ```
8. Commit changes in git and push.
9. Submit pull request on GitHub.


## Unit tests

First, set your terminal directory to this repo's root directory. Next, make sure you have activated the appropriate conda environment for the SustainGym environment you want to test (e.g., `conda activate sustaingym_ev`). Finally, run the unit tests for the desired SustainGym environment:

```bash
python -m unittest -v tests/test_evcharging.py
```


## Building PyPI package

```bash
# create a conda environment with appropriate build tools
conda env update --file env_build.yml --prune
conda activate build

# remove generated pkl files from CogenEnv
rm sustaingym/data/cogen/ambients_data/*.pkl

# clean cached build files
rm -r sustaingym.egg-info
rm -r dist

# build source dist (.sdist) and wheel (.whl) files
python -m build

# upload built files to PyPI
twine upload --repository testpypi dist/*  # for testpypi
twine upload dist/*
```


# Website and documentation

The SustainGym website and documentation is written in the `docs/` folder and built using [Sphinx](https://www.sphinx-doc.org/). The [API reference documentation](https://chrisyeh96.github.io/sustaingym/api/sustaingym/) is automatically generated from class and function docstrings. See the section [coding style guide](#coding-style-guide) below for details.

We use the following Sphinx extensions to build our documentation site:
- [`MyST Parser (myst_parser)`](https://myst-parser.readthedocs.io/): allows using Markdown files instead of ReStructuredText files. See the [MyST Roles and Directives documentation](https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html) for instructions on how to use Sphinx directives in Markdown files.
    - Even though we use Markdown for the documentation files, Python docstrings in SustainGym code still must use ReStructuredText syntax for formatting.
- [Sphinx AutoAPI (`autoapi.extension`)](https://sphinx-autoapi.readthedocs.io/): automatically generates API documentation based on Python docstrings. Unlike [`sphinx.ext.autosummary`](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html), Sphinx AutoAPI uses static code analysis without needing to import Python code to read the docstrings. This is important for SustainGym because the different environments within SustainGym are allowed to have conflicting dependencies.
    - Unfortunately, Sphinx AutoAPI does not properly handle namespace packages yet. See [this GitHub issue](https://github.com/readthedocs/sphinx-autoapi/issues/298). This is why have  `__init__.py` files in every directory in the SustainGym codebase, even though many of these files are empty.
- [`sphinx.ext.napoleon`](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html): allows using Google-style docstrings for documenting functions.
    - By default, the this extension does not let us easily document functions that return multiple variables at once (i.e., a tuple return type). Following [this StackOverflow response](https://stackoverflow.com/a/67177881), we choose to use the same documentation format for function arguments and return values by using the following line in our Sphinx configuration at [`docs/conf.py`](./docs/conf.py).
    ```python
    napoleon_custom_sections = [("Returns", "params_style")]
    ```


## Coding style guide

An example of the required coding style is shown below. It is based on the [Google Python style guide](https://google.github.io/styleguide/pyguide.html). Several points deserve elaboration:
- The arguments of a class's `__init__()` function should be documented in the class docstring, instead of the `__init__()` function docstring. Only use the `__init__()` function docstring to document any implementation details. This is because [Sphinx AutoAPI does not explicitly document the `__init__()` function](https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#confval-autoapi_python_class_content).
- In a class docstring, class attributes should be documented with type information.
- In a function docstring, function arguments do not need to be documented with type information as long as appropriate type annotations are provided in the function signature.
- In a function docstring, function return values should be documented just like arguments.

```python
"""This module implements MyClass."""

class MyClass:
    """Short description of class.

    Longer description of class.
    Can be multi-line.

    Args:
        option: str, description of option

    Attributes:
        first_line: str, first line that gets printed
        num_prints: int, number of times that print() has been called. When
            description is long, indent the subsequent lines.

    Example::

        my_obj = MyClass(option='hello')
        text, num_prints = my_obj.print(name='John')
    """
    def __init__(self, option: str):
        self.first_line = option + ' world!'
        self.num_prints = 0

    def print(self, name: str) -> tuple[str, int]:
        """Prints two lines and returns the printed string.

        Args:
            name: str, name that gets printed

        Returns:
            text: text that was printed
            num_prints: number of times that print() has been called
        """
        text = f'{self.instance_var}\nMy name is {name}!'
        self.num_prints += 1
        print(text)
        return text, self.num_prints
```


## Building website locally with Sphinx

To build the website, you can follow these steps either locally on your computer or in a [GitHub Codespace](https://github.com/features/codespaces).

1. If you are installing dependencies locally, I recommend first creating a conda environment by running the following commands from the repo root folder. If you are using a GitHub Codespace, skip this step.

    ```bash
    # create a new conda environment named "docs", and activate it
    conda create -n docs python=3.11 pip
    conda activate docs
    ```

2. Install the necessary dependencies from `docs/requirements.txt`. From the `docs/` folder:

    ```bash
    # install dependencies needed to build website locally
    pip install -r requirements.txt
    ```

3. Clear out the `docs/_build` directory if it exists, including any hidden folders or files (i.e., dotfiles). This isn't always necessary, but at a minimum, do this every time you change `docs/conf.py`. From the `docs/` folder:

    ```bash
    rm -r _build/* _build/.*
    ```

4. Build the documentation using Sphinx. From the `docs/` folder:

    ```bash
    sphinx-build -b html . _build
    ```

5. Launch a server to view the docs. From the `docs/_build` folder:

    ```bash
    python -m http.server
    ```

   Optionally, change the port:

    ```bash
    python -m http.server <port>
    ```


## Deploying website on GitHub Pages

The SustainGym repo automatically compiles and deploys the website from the `main` branch's `docs/` folder to GitHub Pages whenever the `main` branch is pushed to. This process can also be triggered manually from the "Build documentation" workflow in the Actions tab. The GitHub workflow configuration can be found at [.github/workflows/build_docs.yml](.github/workflows/build_docs.yml).
