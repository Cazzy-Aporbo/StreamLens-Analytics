import pytest
from fastapi.testclient import TestClient

from app import app, runtime_service


@pytest.fixture(scope='module')
def client():
    return TestClient(app)


def test_health(client):
    res = client.get('/api/health')
    assert res.status_code == 200
    assert res.json()['status'] == 'ok'


def test_dashboard_served(client):
    res = client.get('/')
    assert res.status_code == 200
    assert 'Stream' in res.text


def test_results(client):
    res = client.get('/api/results')
    assert res.status_code == 200
    body = res.json()
    assert 'overall_metrics' in body
    assert 'temporal_analysis' in body


def test_overview(client):
    res = client.get('/api/metrics/overview')
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body['gender_parity'] <= 1


def test_characters_filtering(client):
    res = client.get('/api/characters', params={'platform': 'netflix', 'limit': 10})
    assert res.status_code == 200
    records = res.json()
    assert len(records) <= 10
    assert all(r['platform'] == 'netflix' for r in records)


def test_analyze_validates_input(client):
    res = client.post('/api/analyze', params={'n_samples': 10})
    assert res.status_code == 422


def test_analyze_runs(client):
    res = client.post('/api/analyze', params={'n_samples': 200})
    assert res.status_code == 200
    assert 'overall_metrics' in res.json()


def test_write_endpoints_require_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("STREAM_API_KEY", "stream-secret")

    denied = client.post('/api/analyze', params={'n_samples': 200})
    assert denied.status_code == 401

    allowed = client.post(
        '/api/analyze',
        params={'n_samples': 200},
        headers={'X-API-Key': 'stream-secret'},
    )
    assert allowed.status_code == 200


def test_genres(client):
    res = client.get('/api/metrics/genres')
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) > 0
    assert {'genre', 'diversity', 'gender_parity', 'male_lead_share', 'dialogue_gap'} <= set(rows[0])


def test_media_types(client):
    res = client.get('/api/metrics/media')
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) > 0
    assert {'media_type', 'diversity', 'gender_parity', 'avg_sentiment', 'sample_size'} <= set(rows[0])


def test_music_quality_surface(client):
    res = client.get('/api/music/quality')
    assert res.status_code == 200
    body = res.json()
    coverage = body['coverage']
    assert 'publication_year_explicit_share' in coverage
    assert 'publication_year_inferred_share' in coverage


def test_music_timeline_surface(client):
    res = client.get('/api/music/timeline')
    assert res.status_code == 200
    body = res.json()
    assert 'years' in body
    assert 'missing_years' in body
    if body['years']:
        assert body['years'] == list(range(body['years'][0], body['years'][-1] + 1))


def test_platform_media_types(client):
    res = client.get('/api/metrics/platform-media')
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) > 0
    assert {'platform', 'media_type', 'diversity', 'gender_parity', 'avg_screen_time', 'avg_sentiment', 'lead_share', 'sample_size'} <= set(rows[0])


def test_characters_media_type_filter(client):
    res = client.get('/api/characters', params={'media_type': 'film', 'limit': 10})
    assert res.status_code == 200
    records = res.json()
    assert all(r['media_type'] == 'film' for r in records)


def test_bias_library(client):
    res = client.get('/api/bias-library')
    assert res.status_code == 200
    body = res.json()
    assert body['total'] >= 50
    assert all({'id', 'category', 'name', 'definition', 'example', 'why_it_matters', 'measured_here'}
               <= set(b) for b in body['items'])


def test_bias_library_category_filter(client):
    res = client.get('/api/bias-library', params={'category': 'gender'})
    assert res.status_code == 200
    body = res.json()
    assert body['total'] > 0
    assert all(b['category'] == 'gender' for b in body['items'])


def test_bias_dimensions(client):
    res = client.get('/api/metrics/bias')
    assert res.status_code == 200
    body = res.json()
    for key in ('age_bias', 'racial_dialogue_bias', 'sentiment_bias', 'screen_time_bias'):
        assert key in body
    for v in body['sentiment_bias'].values():
        assert 'avg_sentiment' in v and 'deviation_from_mean' in v


def test_network(client):
    res = client.get('/api/metrics/network')
    assert res.status_code == 200
    body = res.json()
    assert body['nodes'] > 0
    assert -1 <= body['gender_homophily'] <= 1


def test_intersectionality(client):
    res = client.get('/api/metrics/intersectionality', params={'limit': 5})
    assert res.status_code == 200
    body = res.json()
    assert len(body['most_underrepresented']) <= 5
    ratios = [g['ratio'] for g in body['most_underrepresented']]
    assert ratios == sorted(ratios)


def test_insights(client):
    res = client.get('/api/insights')
    assert res.status_code == 200
    insights = res.json()
    assert len(insights) > 0
    assert {'category', 'title', 'detail'} <= set(insights[0])


def test_lenses_catalog(client):
    res = client.get('/api/lenses/catalog')
    assert res.status_code == 200
    body = res.json()
    assert len(body['items']) >= 4
    assert {'lens_id', 'display_name', 'description'} <= set(body['items'][0])


def test_lenses_demo_stream(client):
    res = client.get('/api/lenses/demo-stream', params={'limit': 6})
    assert res.status_code == 200
    body = res.json()
    assert body['count'] <= 6
    assert len(body['items']) == body['count']
    if body['items']:
        assert {'record', 'finding_count', 'findings'} <= set(body['items'][0])


def test_media_lab_overview(client):
    res = client.get('/api/media-lab/overview')
    assert res.status_code == 200
    body = res.json()
    assert {'compulsive_usage', 'generative_guard', 'causal_map', 'events'} <= set(body)


def test_media_lab_compulsive_loop(client):
    res = client.get('/api/media-lab/compulsive-loop')
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body['risk_score'] <= 1
    assert body['recommended_friction_ms'] >= 0


def test_media_lab_generative_guard(client):
    res = client.get('/api/media-lab/generative-guard')
    assert res.status_code == 200
    body = res.json()
    assert body['blocked'] is True
    assert body['remaining_nonzero_bytes'] == 0


def test_media_lab_causal_map(client):
    res = client.get('/api/media-lab/causal-map')
    assert res.status_code == 200
    body = res.json()
    assert body['node_count'] > 0
    assert body['edge_count'] > 0


def test_media_lab_insurability(client):
    res = client.get('/api/media-lab/insurability')
    assert res.status_code == 200
    body = res.json()
    assert body['analysis']['posture_band'] in {'clear', 'watch', 'elevated', 'high'}


def test_runtime_review_demo(client):
    res = client.get('/api/runtime/review/demo')
    assert res.status_code == 200
    body = res.json()
    assert body['example']['blocked'] is True


def test_runtime_review_post(client):
    res = client.post('/api/runtime/review', json={
        'prompt': 'Summarize the billing incident briefly.',
        'draft_response': 'Customer contacts: jordan@example.com and +1 202 555 0148.'
    })
    assert res.status_code == 200
    body = res.json()
    assert body['blocked'] is True
    assert 'draft_payload' in body['evidence_sources']


def test_runtime_drift_demo(client):
    res = client.get('/api/runtime/drift/demo')
    assert res.status_code == 200
    body = res.json()
    assert body['analysis']['drift_band'] in {'clear', 'watch', 'elevated', 'severe'}


def test_runtime_drift_post(client):
    res = client.post('/api/runtime/drift', json={
        'turns': [
            'I can share the safe cohort summary.',
            'Customer contacts: jordan@example.com.',
            'I will not share the lyrics.'
        ]
    })
    assert res.status_code == 200
    body = res.json()
    assert body['signals']['turn_count'] == 3


def test_learn(client):
    res = client.get('/api/learn')
    assert res.status_code == 200
    cards = res.json()
    assert len(cards) >= 5
    assert all({'title', 'summary', 'detail', 'try_it'} <= set(c) for c in cards)


def test_data_engineering_surface(client):
    res = client.get('/api/system/data-engineering')
    assert res.status_code == 200
    body = res.json()
    assert {'generated_at', 'operating_model', 'service_levels', 'delivery_surfaces', 'stages', 'lineage', 'reproducibility', 'contracts', 'quality_highlights'} <= set(body)
    assert body['operating_model']['dataset_count'] >= 2
    assert len(body['service_levels']) >= 4
    assert len(body['delivery_surfaces']) >= 3
    assert len(body['stages']) >= 4
    assert len(body['contracts']) >= 2
    assert body['reproducibility']['seed'] == 42
    assert len(body['reproducibility']['default_sample_sizes']) >= 4
    first_contract = body['contracts'][0]
    assert {'dataset_id', 'grain', 'primary_key', 'partition_keys', 'quality_checks', 'schema_profile', 'schema'} <= set(first_contract)
    assert len(first_contract['quality_checks']) >= 3
    assert len(first_contract['schema']) >= 5
    assert first_contract['schema_profile']['column_count'] >= 5


def test_system_integrations_surface(client):
    res = client.get('/api/system/integrations')
    assert res.status_code == 200
    body = res.json()
    assert len(body['available_now']) >= 4
    assert 'commercial_boundary' in body


def test_system_api_contracts_surface(client):
    res = client.get('/api/system/api-contracts')
    assert res.status_code == 200
    body = res.json()
    assert {'route_count', 'write_route_count', 'routes', 'schemas', 'write_surface_policy'} <= set(body)
    assert body['route_count'] >= 40
    assert body['write_surface_policy']['header'] == 'X-API-Key'


def test_system_model_registry_surface(client):
    res = client.get('/api/system/model-registry')
    assert res.status_code == 200
    body = res.json()
    assert {'trained_models', 'statistical_methods', 'deterministic_probes', 'compiled_kernels', 'not_claimed'} <= set(body)
    assert any(model['id'] == 'music_view_predictability_ensemble' for model in body['trained_models'])


def test_system_bias_dynamics_surface(client):
    res = client.get('/api/system/bias-dynamics')
    assert res.status_code == 200
    body = res.json()
    assert {'headline', 'posture_band', 'bias_score', 'dimensions', 'watchpoints', 'signals'} <= set(body)
    assert body['posture_band'] in {'clear', 'watch', 'elevated', 'concentrated'}
    assert 0 <= body['bias_score'] <= 1
    assert len(body['dimensions']) >= 4


def test_system_runtime_ledger_surface(client):
    runtime_service.materialize(sample_size=500, persist_sqlite=True)
    res = client.get('/api/system/runtime/ledger')
    assert res.status_code == 200
    body = res.json()
    assert body['status'] in {'empty', 'verified', 'mixed', 'broken'}
    assert 'items' in body


def test_runtime_metrics_live_surface(client):
    res = client.get('/api/runtime/metrics/live')
    assert res.status_code == 200
    body = res.json()
    assert body['source'] in {'live', 'demonstration'}
    assert body['pressure_band'] in {'open', 'watch', 'narrow', 'compressed'}
    assert 'metrics' in body
    assert 'watchpoints' in body
    assert 'invariants' in body


def test_runtime_invariants_live_surface(client):
    res = client.get('/api/runtime/invariants/live')
    assert res.status_code == 200
    body = res.json()
    assert body['action'] in {'allow', 'review', 'quarantine'}
    assert len(body['invariants']) >= 5


def test_runtime_event_contract_surface(client):
    res = client.get('/api/runtime/events/contract')
    assert res.status_code == 200
    body = res.json()
    assert {'fields', 'batch_limits', 'retention', 'filters', 'sample_event', 'distribution_bus'} <= set(body)
    assert body['batch_limits']['max_events'] >= 100
    assert len(body['filters']) >= 4


def test_runtime_events_demo_seed_and_latest(client):
    seeded = client.post('/api/runtime/events/demo-seed')
    assert seeded.status_code == 200
    seed_body = seeded.json()
    assert seed_body['status'] == 'ok'
    assert seed_body['event_count'] >= 1
    assert 'bus' in seed_body

    latest = client.get('/api/runtime/events/latest', params={'limit': 5})
    assert latest.status_code == 200
    latest_body = latest.json()
    assert latest_body['limit'] == 5
    assert len(latest_body['items']) >= 1

    filtered = client.get('/api/runtime/metrics/live', params={'market': 'PH'})
    assert filtered.status_code == 200
    filtered_body = filtered.json()
    assert filtered_body['filters'] == {'market': 'PH'}


def test_runtime_metrics_live_socket(client):
    client.post('/api/runtime/events/demo-seed')
    with client.websocket_connect('/ws/runtime/metrics/live?market=PH&interval_ms=1000') as websocket:
        body = websocket.receive_json()
    assert body['filters'] == {'market': 'PH'}
    assert body['pressure_band'] in {'open', 'watch', 'narrow', 'compressed'}


def test_governance_surface(client):
    res = client.get('/api/system/governance')
    assert res.status_code == 200
    body = res.json()
    assert {'summary', 'domains', 'questions', 'contribution_paths'} <= set(body)
    assert body['summary']['repository_mode'] == 'independent_open_source_surface'
    assert len(body['domains']) >= 5
    assert any(domain['id'] == 'gdpr_boundary' for domain in body['domains'])


def test_readiness_surface(client):
    res = client.get('/api/system/readiness')
    assert res.status_code == 200
    body = res.json()
    assert {'overall', 'proof_points', 'quickstart', 'adoption_paths', 'commercial_posture'} <= set(body)
    assert body['overall']['level'] in {'pass', 'warn', 'fail'}
    assert body['proof_points']['dataset_count'] >= 2
    assert body['proof_points']['local_ledger_mode'] in {'hash_linked', 'shape_incomplete'}
    assert body['proof_points']['write_endpoint_mode'] in {'protected', 'open_local_default'}
    assert body['proof_points']['entrypoint_count'] >= 3
    assert len(body['quickstart']) >= 3
    assert len(body['adoption_paths']) >= 3
    assert len(body['commercial_posture']['not_claiming_yet']) >= 3


def test_streaming_readiness_surface(client):
    res = client.get('/api/system/streaming-readiness')
    assert res.status_code == 200
    body = res.json()
    assert {'positioning', 'architecture_concerns', 'production_expectations_missing', 'roadmap'} <= set(body)
    assert body['positioning']['maturity_label'] == 'research_foundation_with_operational_signals'
    assert len(body['architecture_concerns']) >= 3
    assert len(body['production_expectations_missing']['runtime_guarantees']) >= 5
    assert len(body['roadmap']['quick_wins']) >= 3


def test_music_theory_surface(client):
    res = client.get('/api/music/theory')
    assert res.status_code == 200
    body = res.json()
    assert body['posture'] in {'measured', 'waiting_for_scores'}
    assert 'coverage' in body
    assert 'pitch_surface' in body
    assert 'claim_boundaries' in body


def test_bias_propagation_surface(client):
    res = client.get('/api/system/bias-propagation')
    assert res.status_code == 200
    body = res.json()
    assert {'stages', 'roles', 'items', 'notes'} <= set(body)
    assert len(body['stages']) >= 6
    assert len(body['roles']) >= 4
    assert len(body['items']) >= 50
    first = body['items'][0]
    assert {'id', 'name', 'category', 'entry_stage', 'propagation_path', 'harm_profile', 'wave_profile', 'role_routes'} <= set(first)
    assert len(first['propagation_path']) >= 2
    assert {'creator', 'operator', 'buyer', 'public'} <= set(first['role_routes'])


def test_trojan_horse_surface(client):
    res = client.get('/api/system/trojan-horse')
    assert res.status_code == 200
    body = res.json()
    assert {'generated_at', 'headline', 'description', 'presets', 'package'} <= set(body)
    assert len(body['presets']) >= 5
    assert body['package']['name'] == '@loopchii/loopchii-lite'


def test_playground_simulate_blocks_pii(client):
    res = client.get('/api/playground/simulate', params={'prompt': 'Use the customer email export and phone list.'})
    assert res.status_code == 200
    body = res.json()
    assert body['blocked'] is True
    assert body['category'] == 'pii'
    assert body['governed_risky_tokens_rendered'] == 0


def test_export_has_download_header(client):
    res = client.get('/api/export')
    assert res.status_code == 200
    assert 'attachment' in res.headers['content-disposition']


def test_advanced_metrics(client):
    res = client.get('/api/metrics/advanced')
    assert res.status_code == 200
    body = res.json()
    assert {'inequality', 'diversity_detail', 'effect_sizes',
            'trend', 'confidence', 'scorecard'} <= set(body)
    assert 0 <= body['inequality']['screen_time']['gini'] <= 1
    assert body['inequality']['screen_time']['lorenz'][0] == {'p': 0.0, 'l': 0.0}


def test_scorecard(client):
    res = client.get('/api/metrics/scorecard')
    assert res.status_code == 200
    body = res.json()
    assert len(body['platforms']) > 0
    for row in body['platforms']:
        assert 0 <= row['overall'] <= 1
        assert row['grade']
    overalls = [p['overall'] for p in body['platforms']]
    assert overalls == sorted(overalls, reverse=True)


def test_simulate_parity_balanced(client):
    res = client.get('/api/simulate/parity', params={'female_ratio': 0.5})
    assert res.status_code == 200
    body = res.json()
    assert body['parity'] == pytest.approx(1.0)
    assert body['grade'] == 'A+'


def test_simulate_parity_extreme(client):
    res = client.get('/api/simulate/parity', params={'female_ratio': 0.0})
    assert res.status_code == 200
    assert res.json()['parity'] == pytest.approx(0.0)


def test_simulate_parity_validates_range(client):
    res = client.get('/api/simulate/parity', params={'female_ratio': 1.5})
    assert res.status_code == 422
