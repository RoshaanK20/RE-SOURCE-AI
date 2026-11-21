from resource_ai.scientific_ui_paths import check_science_inputs


def test_blank_fields_get_helpful_errors():
    issues = check_science_inputs('', '', '', '')
    assert len(issues) == 4
    assert 'Scientific contract' in issues[0]


def test_folder_is_not_json_file(tmp_path):
    src = tmp_path / 'original'; src.mkdir()
    cand = tmp_path / 'candidate'; cand.mkdir()
    ev = tmp_path / 'eval.py'; ev.write_text('print(1)')
    issues = check_science_inputs(str(tmp_path), str(src), str(cand), str(ev))
    assert any('expected a .json file' in msg for msg in issues)


def test_valid_paths(tmp_path):
    src = tmp_path / 'original'; src.mkdir()
    cand = tmp_path / 'candidate'; cand.mkdir()
    ev = tmp_path / 'eval.py'; ev.write_text('print(1)')
    con = tmp_path / 'contract.json'; con.write_text('{}')
    assert check_science_inputs(str(con), str(src), str(cand), str(ev)) == []


def test_evaluator_inside_source_rejected(tmp_path):
    src = tmp_path / 'original'; src.mkdir()
    cand = tmp_path / 'candidate'; cand.mkdir()
    ev = src / 'eval.py'; ev.write_text('print(1)')
    con = tmp_path / 'contract.json'; con.write_text('{}')
    assert any('outside' in msg for msg in check_science_inputs(str(con), str(src), str(cand), str(ev)))
