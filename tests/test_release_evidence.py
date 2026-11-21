from resource_ai.release_evidence import llm_run_record, validation_summary


def test_evidence_identity_and_claims(tmp_path):
    source, reference = tmp_path / 'source.py', tmp_path / 'ref.py'
    source.write_text('print(1)')
    reference.write_text('print(2)')
    data = llm_run_record(case='EXAMPLE', model='qwen2.5-coder:3b',
                          source=source, reference=reference, error='Model missing')
    assert data['run_status'] == 'ERROR'
    assert data['observed_result'] is None
    assert data['source_sha256'] != data['reference_sha256']
    assert data['published_research_reproduction'] is False


def test_validation_summary_does_not_turn_score_into_reproduction():
    summary = validation_summary({'cases': [
        {'dataset': 'wine', 'expected': 'METHOD_CHANGED', 'passed': True, 'score_only_accept': True},
        {'dataset': 'wine', 'expected': 'REPRODUCED', 'passed': True, 'score_only_accept': True}
    ]})
    assert summary['invalid_cases_score_rule_accepted'] == 1
    assert summary['real_research_repository_reproductions'] == 0


def test_qwen_records_reappear_in_dashboard_history(tmp_path):
    from resource_ai.dashboard_logic import recorded_cases
    import json
    p = tmp_path / 'runs' / '2026_qwen_repair' / 'llm_repair_result.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'model_id':'qwen2.5-coder:7b', 'observed_result': {
        'llm_calls': 1, 'final_verdict': 'REPRODUCED'}}))
    rows = recorded_cases(tmp_path)
    assert len(rows) == 1
    assert rows[0]['kind'] == 'llm'
    assert rows[0]['status'] == 'REPRODUCED'
