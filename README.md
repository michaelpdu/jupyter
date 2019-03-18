# How to set up Jupyter Notebook?

In Aliyun ECS, we could set up ourselves Jupyter Notebook as follow:

- Generate Jupyter config and set the password

```
jupyter notebook --generate-config  # generate config

ipython # stary ipython to set password
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
nohup jupyter notebook --allow-root > jupyter.log 2>&1 &
```
