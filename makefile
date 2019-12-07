# Austin Corum
# Requires python and pip to be installed
# Python makefile, to run, type "make" and ENTER into terminal
# ADD * pip install all dependencies *
### Best to run this on a UNIX/Linux environment ###

dropout-figure.png:
	pip install -r requirements.txt | { grep -v "already satisfied" || :; }
	python dropout.py
