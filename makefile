# Austin Corum
# python makefile, to run "make" and ENTER
# ADD * pip install all dependencies *

dropout-figure.png: python dropout.py
	pip install -r requirements.txt | { grep -v "already satisfied" || :; }
