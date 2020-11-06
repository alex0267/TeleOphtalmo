# Introduction

This document details how to use this repository on the `teleophtalmo` GCP VM.


# Configuring SSH

Sometimes, one encounters slowdowns on the VM's Jupyter Lab rendering it close to unusable. For this reason, it may be useful to connect to the VM via SSH. This requires setting up your public key. This sections summarizes the [official documentation](https://cloud.google.com/compute/docs/instances/connecting-to-instance).

We'll first need to install the Cloud SDK, make sure you have Python 3.5 to 3.8 installed locally and download the SDK, extract it and run the installation script:

```sh
curl https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-317.0.0-darwin-x86_64.tar.gz
tar xvzf google-cloud-sdk-317.0.0-darwin-x86_64.tar.gz
./google-cloud-sdk/install.sh
```

You'll now need to initialize the SDK which will log you in:

```sh
./google-cloud-sdk/bin/gcloud init
```

The initialization script will prompt you for a cloud project to use, select `rare-result-248415`. If you are asked for a *compute engine zone*, select `europe-west1-b`.

You should now be able to to SSH into the VM by running:

```sh
 ./google-cloud-sdk/bin/gcloud compute ssh teleopthalmo
```

Note that if you open a new terminal after the SDK installation process, you can simply run `gcloud` rather that the full path to the binary as done above.

# Configuring Git

Once you are SSH'ed into the VM, you'll need to clone the TeleOpthalmo repository, though you will need to configure you SSH key. The simplest and most secure way is to create a new key pair rather than `scp`'ing your own:

```sh
ssh-keygen
cat ~/.ssh/id_rsa.pub
```

You can now copy the output from the `cat` command and create a new SSH key in your Github profile. You'll now be able to clone the TeleOphtalmo repository:

```sh
git clone git@github.com:WeWyse/TeleOphtalmo.git
```

## Commiting

In order to commit to the repository from the VM, you'll need to setup your Git profile:

```sh
git config --global user.email "YOUR_EMAIL"
git config --global user.name "FIRST_NAME LAST_NAME"
```

# Setting up the Python environment

Once your are in the `TeleOphtalmo` directory, create a Python virtual environment, activate it and install the dependencies:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You should now be able launch a Python REPL and import the models. Make sure to be in the `module` folder or else the `import` statements will fail:

```sh
cd module
```

```python
>>> import branch1
>>> import MRCNN
```

# Troubleshooting

Please contact me via Slack or tbinetruy@wewyse.com if you run into touble ;)
