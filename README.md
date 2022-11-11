# sustaingym
Reinforcement learning environments for sustainability applications

# Install Jekyll

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

# Build

**Build locally**: `jekyll build [options]`

**Serve locally**: `jekyll serve [options]`

Options
- `--baseurl /some_path`: add this if hosting this site at some non-root domain (e.g., `*.com/some_path`)
- `--drafts`: to show drafts among the latest posts
- `--force_polling`: use this when running Jekyll on WSL to enable auto-regeneration.