# 🛑 This branch is no longer active. 🛑

**As of 2023-09-13, the SustainGym website is no longer built from this branch. Instead, it is built from the `docs/` folder on the `main` branch. The `gh_pages` branch was renamed `gh_pages_ARCHIVE` and is only kept around as a historical artifact. To edit the website, please see the [CONTRIBUTING guide](https://github.com/chrisyeh96/sustaingym/blob/main/CONTRIBUTING.md) on the `main` branch.**

The rest of this README remains as-is from the previous website.


# SustainGym Website

The SustainGym website is hosted via [GitHub Pages](https://pages.github.com/). There are two suggested ways to edit this website:

1. Recommended: edit the website online, using [GitHub Codespaces](https://github.com/features/codespaces).
2. Alternative: edit the website locally.


## Edit website using GitHub Codespaces

1. Open the `gh_pages` branch in a Codespace. The `gh_pages` branch has been configured (via [`devcontainer.json`](.devcontainer/devcontainer.json)) so that the default Codespace will automatically have all necessary dependencies installed.
2. Edit the website files.
3. To preview the website, run the following in the terminal within the Codespace:
    ```bash
    jekyll serve --baseurl ""
    ```
   You should see a popup with an "Open in Browser" button. Click on the button to view the compiled website. Any changes you make to the Markdown website files should automatically get reflected when you refresh the webpage in your browser.
4. Once you are happy with the changes, commit the changes to a new branch and create a pull request to the `gh_pages` branch.


## Edit website locally

This section assumes that you are running a Debian-based Linux distribution (e.g., Ubuntu). If you use Windows, you may also use Ubuntu running in Windows Subsystem for Linux (WSL).

**Clone `gh_pages` branch**

The following commands clones the `gh_pages` branch into a new folder called `sustaingym_website`. It then creates a new branch called `gh_pages_edit`.

```bash
git clone https://github.com/chrisyeh96/sustaingym.git --branch gh_pages --single-branch sustaingym_website
git checkout -b gh_pages_edit
```

**Install Jekyll**

I find it best to use the conda package manager to install ruby, then use ruby to install the `github-pages` gem, which contains the set of gems (including Jekyll) used by GitHub Pages itself to compile each site. If conda is not already installed, [install conda first](https://docs.conda.io/en/latest/miniconda.html). Then, run the following commands:

```bash
conda env update -f env.yml --prune
conda activate ruby

# Install/update to the latest version of github-pages gem
bundle install
bundle clean --force
```

If the `bundle install` command causes an error, try the following commands and then try `bundle install` again.

```bash
sudo apt update  # update package index
sudo apt upgrade build-essential  # compilation tools
```

**Serve**

```bash
jekyll serve
```

The command above compiles the Markdown website files into HTML files and serves a local copy of the website at [http://localhost:4000](http://localhost:4000). Any changes you make to the Markdown files should automatically get reflected when you refresh the webpage in your browser. If you are running on WSL, you may need to add the `--force_polling` option to enable auto-regeneration.

**Commit and pull-request**

Once you are happy with the changes, commit the changes. Push your commits, then create a pull request to the `gh_pages` branch.
