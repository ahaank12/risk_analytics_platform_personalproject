.PHONY: demo setup ingest seed run

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

ingest:
	. .venv/bin/activate && python scripts/ingest_market_data.py --start 2018-01-01

seed:
	. .venv/bin/activate && python scripts/seed_portfolio.py

run:
	. .venv/bin/activate && streamlit run app.py

demo:
	./run_demo.sh
