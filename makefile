# Austin Corum
# python makefile, to run "make" and ENTER
# ADD * pip install all dependencies *
# best to run this on a UNIX/Linux environment

dropout-figure.png:
	pip install -r requirements.txt | { grep -v "already satisfied" || :; }
	python dropout.py
