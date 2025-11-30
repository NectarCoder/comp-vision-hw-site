# Computer Vision Homework Site

This repo contains the code and materials for the Computer Vision Homework Site.  

## Setup venv
This website will not work without the required python packages. Setup a virtual environment like so:  

```bash
# From project root
./setup_venv/setup_venv.sh #for unix systems
```

```ps1
# From project root
.\setup_venv\setup_venv.ps1 #for windows
```

## Launch site in debug
After setting up the venv, you can launch the site in debug mode like so:

```bash
# From project root
python3 app.py
```

## Launch site in production
Setup gunicorn (unix) or waitress (windows). Instructions for that can be found on the internet.  