# How to set up Jupyter Notebook?

## Jupyter Notebook

In Aliyun ECS, we could set up ourselves Jupyter Notebook as follow:

- Generate Jupyter config and set the password

```
jupyter notebook --generate-config  # generate config

ipython # start ipython to set password
from notebook.auth import passwd
passwd() # set password here
# after setting your password, copy the sha1 string to following config.py
```

- Modify Jupyter config

```
vim ~/.jupyter/jupyter_notebook_config.py

c.NotebookApp.ip = '*'
c.NotebookApp.password = 'sha1:65xxxx'
c.NotebookApp.password_required = True
c.NotebookApp.open_browser = False
c.NotebookApp.port = 6484 # change to your port
```

- Start Jupyter Notebook in background

```
nohup jupyter notebook --allow-root > /tmp/jupyter.log 2>&1 &
```

- Access Jupyter server in your desktop browser

## Jupyter Lab

Another way to set up Jupyter Environment is to us Jupyter Lab, it's simpler than above Notebook.

```
jupyter lab --ip 10.15.33.211 --port 8890 --app-dir ~/github/jupyter/
```

Note: In console mode, browser cannot open directly, so you need to quit by `q` command.

Then copy URL to your browser, and access jupyter lab.