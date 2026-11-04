.PHONY: setup validate gui demo inspect clean

setup:
	./scripts/setup.sh

validate:
	./scripts/validate_installation.sh

gui:
	./scripts/run_gui.sh

demo:
	./scripts/run_demo.sh

inspect:
	uv run gem-learning-engine inspect-model --config configs/demo.yaml --output-dir outputs/demo_inspection

clean:
	rm -rf outputs/demo_exact_fba outputs/demo_inspection outputs/demo_preflight.json outputs/installation_validation
