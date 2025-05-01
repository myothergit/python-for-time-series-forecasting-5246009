if [ -f requirements.txt ]; then
  pip install --user -r requirements.txt
fi

python -m ipykernel install --user
mkdir -p ~/.ipython/profile_default/startup && cp .devcontainer/pandas-startup.py ~/.ipython/profile_default/startup/00-pandas-options.py