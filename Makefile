PYTHONPATH ?= src
PYTEST ?= pytest

.PHONY: test-repro test-governance test-mlflow test-batch test-all-local repro-audit evidence-bundle dvc-template-smoke acceptance-bundle

test-repro:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) -q \
		tests/test_deterministic_artifact_digest.py \
		tests/test_repro_audit_snapshot.py \
		tests/test_stage_trace_writer.py \
		tests/test_validate_only_and_determinism.py

test-governance:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) -q \
		tests/test_batch_artifact_validation.py \
		tests/test_batch_runner_dvc_template.py

test-mlflow:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) -q \
		tests/test_mlflow_local_logging.py \
		tests/test_batch_mlflow_logging.py

test-batch:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) -q \
		tests/test_batch_runner_dvc_template.py \
		tests/test_validate_only_and_determinism.py \
		tests/test_batch_artifact_validation.py

test-all-local:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) -q \
		tests/test_stage_trace_writer.py \
		tests/test_llama_summary_parser.py \
		tests/test_batch_artifact_validation.py \
		tests/test_validate_only_and_determinism.py \
		tests/test_batch_runner_dvc_template.py \
		tests/test_deterministic_artifact_digest.py \
		tests/test_repro_audit_snapshot.py \
		tests/test_mlflow_local_logging.py \
		tests/test_batch_mlflow_logging.py

repro-audit:
	python3 scripts/repro_audit.py --config configs/pipeline.nemo.llama.yaml --meeting-id ES2005a

evidence-bundle:
	python3 scripts/generate_acceptance_evidence_bundle.py --meeting-id ES2005a --include-batch-runs

dvc-template-smoke:
	python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.yaml --meeting-id ES2005a --validate-only --dvc-template single

acceptance-bundle:
	python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.strict_offline.yaml --meeting-id ES2005a --validate-only --dvc-template single
	python3 scripts/repro_audit.py --config configs/pipeline.nemo.llama.strict_offline.yaml --meeting-id ES2005a
	python3 scripts/generate_acceptance_evidence_bundle.py --meeting-id ES2005a --include-batch-runs
