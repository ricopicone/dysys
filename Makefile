.PHONY: pypi conda dev

pypi:
	poetry lock
	poetry build
	poetry publish

conda: environment.yaml

environment.yaml:
	conda env export | grep -v "^prefix: " > environment.yml

dev: # Develop inside a conda environment with poetry (only install packages with poetry!)
	conda create -n dysys python=3.11
	conda activate dysys
	conda install poetry