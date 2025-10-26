.PHONY: setup lint test train report

setup:
\tpip install -r requirements.txt
\tpip install ruff black pytest

lint:
\truff check .
\tblack --check .

test:
\tpytest -q

train:
\tpython -m src.train --train

report:
\tpython -m src.train --report
