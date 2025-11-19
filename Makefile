all: build install clear
	echo "built"

build:
	python -m build > /dev/null

install: build
	pip install --upgrade --no-deps --force-reinstall dist/currybo-0.1.0-py3-none-any.whl > /dev/null

clear:
	rm -f runs/*.pt
