.PHONY: install notebook tensorboard clean

install:
	pip install -r requirements.txt

notebook:
	jupyter notebook

tensorboard:
	tensorboard --logdir=logs/

clean:
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
