"""RE:SOURCE AI | Research Recovery Console.

Launch: streamlit run app.py
Public GitHub repositories are downloaded for STATIC INSPECTION ONLY.
Reviewed and trusted projects can use a dedicated local Python venv, or restricted Docker.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from resource_ai import __version__
from resource_ai.batch import read_research_list, scan_research_list
from resource_ai.dashboard_logic import (
    available_builtin_cases, collect_evidence, recorded_cases,
    save_report, status_summary, unique_run_dir,
    uploaded_list_path, validate_ui_research_list,
)
from resource_ai.intake import github_source, intake_source
from resource_ai.llm import OllamaLLM, probe_ollama
from resource_ai.release_evidence import llm_run_record, validation_summary
from resource_ai.framework_trial import check_docker, run_trial
from resource_ai.local_runtime import check_local_python
from resource_ai.science import verify_science, as_markdown
from resource_ai.scientific_ui_paths import check_science_inputs

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
BUILTIN_LIST = ROOT / 'corpus' / 'PAPERS_TO_SCAN.md'

st.set_page_config(page_title='RE:SOURCE AI | Research Recovery Console',
                   page_icon='🧪', layout='wide', initial_sidebar_state='expanded')
st.markdown('''
<style>
  .block-container {max-width: 1230px; padding-top: 1.65rem; padding-bottom: 3rem}
  [data-testid="stSidebar"] {background: #101d2a;}
  [data-testid="stSidebar"] * {color: #e7edf5 !important;}
  .rs-kicker {font-size: .82rem; letter-spacing: .16em; text-transform: uppercase;
              color: #3da5a5; font-weight: 750; margin-bottom: .2rem}
  .rs-head {font-size: 2.2rem; font-weight: 780; letter-spacing: -.045em;
            line-height: 1.2; margin-bottom: .6rem}
  .rs-lead {color: #6d7c8a; font-size: 1rem; margin-bottom: 1.35rem}
  .rs-note {padding: 12px 15px; background: rgba(28, 151, 146, .075);
            border-left: 3px solid #189690; border-radius: 5px; margin: 12px 0}
  div[data-testid="stMetric"] {background: rgba(124,138,152,.055); border: 1px solid rgba(124,138,152,.14);
     padding: 13px 16px; border-radius: 10px}
  .rs-footer {color: #81909c; font-size: .8rem; margin-top: 30px;
              border-top:1px solid rgba(120,130,140,.2); padding-top:10px}
</style>''', unsafe_allow_html=True)


def title(eyebrow: str, heading: str, subtitle: str):
    st.markdown(f'<div class="rs-kicker">{eyebrow}</div><div class="rs-head">{heading}</div>'
                f'<div class="rs-lead">{subtitle}</div>', unsafe_allow_html=True)


def notices():
    st.info('**Source inspection is not scientific reproduction.** GitHub projects are never executed '
            'by the scanner. Recovery requires a trusted reference, a supported framework, '
            'and an isolated execution environment.')


def inventory_view(inventory: dict):
    a, b, c, d = st.columns(4)
    a.metric('Python scripts', len(inventory.get('python_files', [])))
    b.metric('Notebooks', len(inventory.get('notebooks', [])))
    c.metric('Dependency files', len(inventory.get('dependency_files', [])))
    d.metric('Compatibility leads', len(inventory.get('compatibility_hints', [])))
    terms = inventory.get('detected_terms', [])
    if terms:
        st.caption('Detected framework / model terms: ' + ', '.join(terms))
    leads = inventory.get('compatibility_hints') or []
    if leads:
        st.subheader('Potential compatibility issues')
        st.caption('Static hypotheses. These do not establish that the original code is broken.')
        st.dataframe(pd.DataFrame([{'File': x.get('file'), 'Possible issue': x.get('potential_issue'),
                                    'Suggested check': x.get('suggested_check'),
                                    'Scientific warning': x.get('scientific_warning')}
                                   for x in leads]), use_container_width=True, hide_index=True)
    else:
        st.caption('No curated compatibility patterns matched the inspected source.')
    with st.expander('Source inventory', expanded=False):
        st.write('**Python files:**', inventory.get('python_files') or 'None')
        st.write('**Notebooks:**', inventory.get('notebooks') or 'None')
        st.write('**Dependency files:**', inventory.get('dependency_files') or 'None')
        st.write('**Static parse warnings:**', inventory.get('static_parse_issues') or 'None')


def report_buttons(case_dir: Path, prefix: str):
    for label, fname, mime in [
        ('Download inspection report (.md)', 'CASE_REPORT.md', 'text/markdown'),
        ('Download evidence record (.json)', 'case.json', 'application/json')]:
        p = case_dir / fname
        if p.is_file():
            st.download_button(label, data=p.read_bytes(), file_name=f'{prefix}_{fname}',
                               mime=mime, key=f'{prefix}_{fname}_{str(case_dir)}')


def show_case(result: dict):
    st.divider()
    st.subheader('Inspection results')
    st.caption(f"Status: {result.get('status', '?')}  |  Source revision: "
               f"{result.get('repository_commit') or 'not confirmed'}")
    if result.get('status') == 'LINK_RECORDED':
        st.warning('Only the link was recorded. Enable “Fetch code” and run again to inspect actual source.')
    else:
        inventory_view(result.get('inventory') or {})
    st.info('**Scientific status: NOT VERIFIED.** No external experiment was executed or reproduced.')
    if result.get('status') in ('REMOTE_INSPECTED', 'LOCAL_INSPECTED'):
        st.caption('Local source folder for optional reviewed Docker trial: ' + str(result.get('project_dir', 'unknown')))
    case_dir = Path(result['case_dir'])
    report_buttons(case_dir, case_dir.name)


def single_repo():
    title('01 / Inspect', 'Analyze a research repository',
          'Paste a GitHub link. RE:SOURCE reads source files and reports possible compatibility issues.')
    with st.form('single_repo'):
        repository_url = st.text_input('GitHub repository or supported source-file link',
                                       placeholder='https://github.com/owner/research-project')
        paper_url = st.text_input('Research paper URL (optional)', placeholder='https://arxiv.org/abs/...')
        fetch = st.checkbox('Fetch source files for inspection', value=True,
                            help='Downloads public GitHub code for reading. Never executes it.')
        submitted = st.form_submit_button('Analyze repository', type='primary', use_container_width=True)
    if submitted:
        try:
            github_source(repository_url)
            if paper_url and not paper_url.startswith('https://'):
                raise ValueError('Paper URL must use HTTPS')
            run = unique_run_dir(RUNS, 'repository')
            with st.status('Inspecting source', expanded=True) as progress:
                st.write('Validating GitHub address')
                st.write('Downloading and reading code' if fetch else 'Recording repository link only')
                result = intake_source(repository_url, run / 'cases', fetch=fetch, paper=paper_url)
                st.write('Creating source inventory and evidence files')
                progress.update(label='Inspection completed', state='complete', expanded=False)
            st.session_state['last_single'] = result
        except Exception as exc:
            st.error(f'Inspection was not completed: {type(exc).__name__}: {exc}')
    if st.session_state.get('last_single'):
        show_case(st.session_state['last_single'])


def research_queue():
    title('02 / Batch', 'Research queue',
          'One Markdown or CSV list, multiple paper-to-code links. Process them in order and keep individual reports.')
    options = st.radio('Source list', ['Included five-paper candidate list', 'Upload my own list'], horizontal=True)
    uploaded = None
    if options == 'Upload my own list':
        uploaded = st.file_uploader('Upload CSV, Markdown, TXT or JSON', type=['csv', 'md', 'txt', 'json'])
    elif BUILTIN_LIST.is_file():
        st.caption(f'Using {BUILTIN_LIST.relative_to(ROOT)}. The listed repositories are **candidates**, not confirmed failures.')
    queue_source = BUILTIN_LIST if options.startswith('Included') else None
    if uploaded is not None:
        try:
            # Upload is not persisted until a run starts.
            queue_source = uploaded_list_path(uploaded.name, uploaded.getvalue(),
                                              RUNS / '_uploaded_queues' / 'preview')
        except Exception as exc:
            st.error(str(exc))
    entries = []
    if queue_source:
        try:
            entries = validate_ui_research_list(queue_source)
            st.dataframe(pd.DataFrame([{'ID': e.case_id, 'Paper': e.title,
                                        'GitHub': e.repository_url} for e in entries]),
                         hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f'Invalid list: {exc}')
    fetch = st.checkbox('Download and inspect each repository', value=True, key='batch_fetch',
                        help='Reads available code but never executes it. Needs GitHub access.')
    pressed = st.button('Start batch analysis', type='primary', disabled=not entries, use_container_width=True)
    if pressed:
        run = unique_run_dir(RUNS, 'batch')
        source_copy = run / ('research_links' + Path(queue_source).suffix)
        source_copy.write_bytes(Path(queue_source).read_bytes())
        bar = st.progress(0, text='Preparing research queue')
        activity = st.empty()
        def on_progress(index, total, entry):
            bar.progress(int(index / total * 100), text=f'Analyzed {index}/{total} repositories')
            activity.info(f"{entry['case_id']} · {entry['status']} · "
                          f"{entry['python_files']} Python files · {entry['compatibility_leads']} possible leads")
        try:
            result = scan_research_list(source_copy, run / 'reports', fetch=fetch, on_progress=on_progress)
            st.session_state['last_batch_dir'] = str(run / 'reports')
            st.session_state['last_batch'] = result
            st.success('Batch complete. All available case reports have been saved.')
        except Exception as exc:
            st.error(f'Batch failed: {type(exc).__name__}: {exc}')
    summary = st.session_state.get('last_batch')
    if summary:
        st.divider()
        st.subheader('Batch findings')
        stats = status_summary(summary.get('entries', []))
        one, two, three, four = st.columns(4)
        one.metric('Research links', stats['repositories'])
        two.metric('Source projects read', stats['sources_inspected'])
        three.metric('Python files', stats['python_files'])
        four.metric('Potential issues', stats['compatibility_leads'])
        table = pd.DataFrame(summary['entries'])
        columns = ['case_id','title','status','python_files','notebooks','compatibility_leads','error']
        st.dataframe(table[[c for c in columns if c in table]], hide_index=True, use_container_width=True)
        if not table.empty:
            st.markdown('**Potential compatibility leads by project**')
            graph = table[['case_id','compatibility_leads']].copy().set_index('case_id')
            st.bar_chart(graph)
        st.caption('Charts show static inspection observations only, not repair success or scientific accuracy.')
        folder = Path(st.session_state['last_batch_dir'])
        a, b, c = st.columns(3)
        for col, filename, text, mime in [
            (a, 'BATCH_REPORT.md', 'Download Markdown report', 'text/markdown'),
            (b, 'batch_summary.csv', 'Download CSV results', 'text/csv'),
            (c, 'batch_summary.json', 'Download JSON evidence', 'application/json')]:
            p = folder / filename
            if p.exists():
                col.download_button(text, p.read_bytes(), file_name=filename, mime=mime)
        try:
            payload = collect_evidence(folder)
            st.download_button('Download all inspection reports (ZIP)', payload,
                               file_name='RE_SOURCE_AI_Research_Inspection_Reports.zip',
                               mime='application/zip', use_container_width=True)
        except ValueError as exc:
            st.caption(str(exc))
        st.info('**Scientific status: NOT VERIFIED.** These repositories have not undergone reproducibility testing.')



def controlled_demo():
    title('03 / Controlled Recovery', 'Watch the repair pipeline',
          'Run the bundled example on a real Scikit-learn dataset, then examine the repair evidence.')
    notices()
    st.markdown('### Demonstration A · Repeatable scripted repair')
    st.write('The bundled Random Forest regressor uses the real **Diabetes** dataset. '
             'The compatibility error is constructed, and the repair is predefined. '
             'This demonstrates execution → patch → verifier, **not autonomous research recovery**.')
    if st.button('Run the controlled recovery demonstration', type='primary', use_container_width=True):
        from scripts.official_first_test import main as run_demo
        folder = unique_run_dir(RUNS, 'scripted_repair')
        path = folder / 'controlled_result.json'
        with st.spinner('Running reference, broken case and scripted repair...'):
            try:
                code = run_demo(['--out', str(path)])
                obj = json.loads(path.read_text(encoding='utf-8'))
                st.session_state['last_demo'] = obj
                st.session_state['last_demo_path'] = str(path)
                if code == 0:
                    st.success('Controlled pipeline completed successfully')
                else:
                    st.error('Controlled pipeline did not meet its acceptance checks')
            except Exception as exc:
                st.error(f'Demonstration failed: {type(exc).__name__}: {exc}')
    obj = st.session_state.get('last_demo')
    if obj:
        a,b,c = st.columns(3)
        a.metric('Original runs', 'Yes' if obj.get('reference_execution_ok') else 'No')
        b.metric('Broken fails', 'Yes' if obj.get('broken_execution_ok') is False else 'No')
        c.metric('Repair executes', 'Yes' if obj.get('repaired_execution_ok') else 'No')
        st.markdown('**Recorded outcome:** ' + str(obj.get('repaired_verdict', 'UNKNOWN')))
        measured = obj.get('observed_metric') or {}
        if isinstance(measured.get('reference'), (int, float)) and isinstance(measured.get('candidate'), (int, float)):
            st.caption('Observed metric: ' + str(measured.get('reference_name', 'metric')) +
                       ' (scripted controlled repair; not a Qwen result)')
            values = pd.DataFrame([
                {'run': 'Original reference', 'measured_value': measured['reference']},
                {'run': 'Scripted repaired', 'measured_value': measured['candidate']}
            ]).set_index('run')
            st.bar_chart(values)
        with st.expander('View exact code difference', expanded=True):
            st.code(obj.get('diff') or 'No recorded patch', language='diff')
        st.download_button('Download controlled test result', json.dumps(obj, indent=2),
                           file_name='controlled_result.json', mime='application/json')
    st.divider()
    st.markdown('### Demonstration B · Qwen repair with Ollama')
    st.caption('This calls an actual local model on a trusted bundled benchmark case. '
               'It does not run downloaded GitHub code or establish published-paper reproduction.')
    cases = available_builtin_cases(ROOT)
    model_name = st.text_input('Installed Ollama model tag', value='qwen2.5-coder:7b',
                               help='Use the exact name displayed by ollama list. For less RAM, try qwen2.5-coder:3b.')
    import re
    model_valid = bool(re.fullmatch(r'[A-Za-z0-9._:/-]{1,80}', model_name))
    if not model_valid:
        st.error('Enter a valid model tag with letters, digits, dots, dashes, colons or slashes.')
    health = probe_ollama(model_name) if model_valid else {'model_installed': False, 'error': 'Invalid model tag'}
    if health.get('model_installed'):
        st.success(f"Ollama detected the exact model: {model_name}. Inference has not yet been tested.")
    else:
        st.warning(health.get('error') or 'Ollama unavailable')
        st.code('ollama serve\nollama pull ' + (model_name if model_valid else 'qwen2.5-coder:7b'),
                language='bash')
    if st.button('Test Qwen generation', disabled=not health.get('model_installed')):
        with st.spinner('Asking the local model for a short completion...'):
            health = probe_ollama(model_name, smoke=True, timeout=120)
        st.session_state['qwen_health'] = health
    if st.session_state.get('qwen_health'):
        st.json(st.session_state['qwen_health'], expanded=False)
        if st.session_state['qwen_health'].get('completion_ok'):
            st.success('A live Ollama completion was returned.')
        elif st.session_state['qwen_health'].get('completion_tested'):
            st.error('No usable model completion. See the diagnostic record.')
    case = st.selectbox('Bundled compatibility case', cases,
                        index=cases.index('RF02_max_features_auto_reg')
                        if 'RF02_max_features_auto_reg' in cases else 0)
    iterations = st.slider('Maximum Qwen repair attempts', 1, 4, 2)
    context = st.select_slider('Model context tokens', options=[2048, 4096, 8192], value=4096)
    st.caption('Before an official evaluation, keep the model, environment and attempt limit fixed.')
    if st.button('Run actual Qwen repair', type='primary',
                 disabled=not health.get('model_installed') or not cases, use_container_width=True):
        from resource_ai.agent import repair
        from resource_ai.runner import run_script
        src = ROOT / 'benchmark' / 'cases' / case
        ref_file = src / 'reference' / 'experiment.py'
        damaged = src / 'broken' / 'experiment.py'
        folder = unique_run_dir(RUNS, 'qwen_repair')
        record = None
        log_lines = []
        with st.status('Running controlled Qwen repair', expanded=True) as progress:
            try:
                st.write('Executing the original reference')
                reference = run_script(ref_file, timeout=120)
                if not reference.ok or not reference.trace:
                    raise RuntimeError('Original reference failed or contains no trace: ' + reference.tail(8))
                st.write(f'Calling actual Ollama model: {model_name}')
                def activity(*parts):
                    message = ' '.join(map(str, parts))
                    log_lines.append(message)
                    st.write(message)
                outcome = repair(src / 'broken', reference.trace,
                                 OllamaLLM(model_name, timeout=300, num_ctx=context),
                                 mode='guarded+kb', workdir=folder / 'work',
                                 max_iters=iterations, timeout=120, log=activity)
                record = llm_run_record(case=case, model=model_name, source=damaged,
                                        reference=ref_file, result=outcome.to_dict(), health=health)
                record['agent_activity'] = log_lines
                record['model_configuration'] = {'temperature': 0.0, 'seed': 0, 'num_ctx': context, 'max_iters': iterations}
                progress.update(label='Qwen attempt recorded', state='complete', expanded=False)
            except Exception as exc:
                record = llm_run_record(case=case, model=model_name, source=damaged,
                                        reference=ref_file,
                                        error=f'{type(exc).__name__}: {exc}', health=health)
                record['agent_activity'] = log_lines
                record['model_configuration'] = {'temperature': 0.0, 'seed': 0, 'num_ctx': context, 'max_iters': iterations}
                progress.update(label='Qwen attempt failed; diagnostic saved',
                                state='error', expanded=True)
            finally:
                saved = save_report(record, folder / 'llm_repair_result.json')
                st.session_state['last_qwen'] = record
                st.session_state['last_qwen_report'] = str(saved)
    record = st.session_state.get('last_qwen')
    if record:
        result = record.get('observed_result') or {}
        st.subheader('Observed LLM result')
        if record['run_status'] == 'ERROR':
            st.error(record.get('error'))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric('Repair attempts', result.get('iterations', 0))
            m2.metric('Model calls', result.get('llm_calls', 0))
            m3.metric('Verifier verdict', result.get('final_verdict', 'UNKNOWN'))
            st.caption('These measurements came from this local model run, not a predefined patch.')
            with st.expander('Original versus repaired code', expanded=True):
                st.code(result.get('diff') or 'No modifications recorded', language='diff')
            with st.expander('Model output and patch feedback'):
                for step in result.get('steps', []):
                    st.write(f"Iteration {step.get('iteration')} · {step.get('action')}: "
                             f"{step.get('detail', '')[:1000]}")
        if st.session_state.get('last_qwen_report'):
            st.download_button('Download Qwen evidence JSON', json.dumps(record, indent=2),
                               file_name='llm_repair_result.json', mime='application/json')
        st.caption('The verifier is instrumentation-based and cannot independently certify '
                   'scientific equivalence of arbitrary Python programs.')



def research_trial():
    title('03 / Scientific experiment', 'Cross-framework research trial',
          'PyTorch, TensorFlow, Scikit-learn, JAX and other Python projects: execution and scientific integrity are separate checks.')
    st.warning('**Never run unknown GitHub code directly on your Windows computer.** '
               'Docker remains the safer choice for unfamiliar repositories. Local Python '
               'is ONLY for projects you have independently reviewed and trust. A venv is not a sandbox.')
    docker = check_docker()
    if docker['available']:
        st.success('Docker is available for restricted container experiments.')
    else:
        st.info('Docker is optional and currently unavailable: ' + str(docker['detail']))
    st.markdown('### A. Run a reviewed research entry point')
    st.caption('Enter the inspected project folder, choose a Python entry point, and select an execution environment.')
    backend_choice = st.radio('Execution environment', ['Local Python (trusted only)', 'Docker (restricted)'],
                              horizontal=True, key='trial_execution_backend')
    local_mode = backend_choice.startswith('Local')
    with st.form('cross_framework_trial'):
        project = st.text_input('Reviewed local project directory', placeholder='runs/repository.../cases/.../project')
        entry = st.text_input('Python entry point, relative to project root', value='train.py')
        if local_mode:
            local_python = st.text_input('Dedicated research venv Python executable',
                                         placeholder=r'C:\research_env\Scripts\python.exe',
                                         help='Create a separate environment for trusted research code. Never point to the app .venv.')
            image = ''
            st.caption('A local venv is NOT isolated from your files, installed software, or network.')
        else:
            image = st.text_input('Locally available Docker image', value='python:3.11-slim',
                                  help='Use an image whose libraries and provenance you have reviewed.')
            local_python = ''
        timeout = st.number_input('Execution timeout (seconds)', min_value=5, max_value=600, value=90, step=5)
        repair_mode = st.checkbox('Ask installed Qwen for compatibility repair' +
                                  (' (local: proposal only; not auto-executed)' if local_mode else ' after failure'))
        model = st.text_input('Ollama model tag (if Qwen enabled)', value='qwen2.5-coder:7b')
        reviewed = st.checkbox('I reviewed this project, its dependencies, and the selected execution environment')
        trusted_local = (st.checkbox('I understand local Python can access my computer and I fully trust this code')
                         if local_mode else False)
        go = st.form_submit_button('Run controlled research trial',
                                    disabled=not local_mode and not docker['available'])
    if go:
        if not reviewed or (local_mode and not trusted_local):
            st.error('Review the project and explicitly confirm the risks before execution.')
        elif local_mode and not check_local_python(local_python)['available']:
            st.error(check_local_python(local_python)['detail'])
        else:
            try:
                llm = None
                if repair_mode:
                    state = probe_ollama(model, smoke=True, timeout=120)
                    if not state.get('completion_ok'):
                        st.warning('Qwen is unavailable (' + str(state.get('error')) + '). '
                                   'Recording the original execution only; no AI repair will be attempted.')
                    else:
                        llm = OllamaLLM(model)
                output = unique_run_dir(RUNS, 'paper_trial')
                with st.spinner('Running reviewed project, recording execution outcomes and proposed changes...'):
                    result = run_trial(project, entry, output, image=image, timeout=int(timeout),
                                       llm=llm, max_retries=2 if llm is not None else 0,
                                       backend='local' if local_mode else 'docker',
                                       python_executable=local_python if local_mode else None,
                                       allow_trusted_local=local_mode and trusted_local)
                st.session_state['science_trial_result'] = result
                st.session_state['science_trial_path'] = str(output)
                st.session_state['science_trial_original'] = project
                st.success('Execution evidence saved. Scientific reproduction requires separate review.')
            except Exception as exc:
                st.error(f'Trial not completed: {type(exc).__name__}: {exc}')
    if st.session_state.get('science_trial_result'):
        trial = st.session_state['science_trial_result']
        a,b,c = st.columns(3)
        a.metric('Initial execution', 'OK' if trial['initial_execution']['ok'] else 'FAILED')
        b.metric('Final execution', 'OK' if trial['final_execution']['ok'] else 'FAILED')
        c.metric('Repair attempts', len(trial['attempts']))
        if trial.get('outcome') == 'ENVIRONMENT_BLOCKED':
            st.error('Environment issue: ' + str(trial.get('environment_issue')) + '. '
                     'Configure a compatible, reviewed Python environment before asking Qwen to edit source code.')
        if trial.get('outcome') == 'PATCH_REVIEW_REQUIRED':
            st.warning('Qwen proposed a patch, but **local Python never executes unreviewed generated edits**. '
                       'Review the proposed changes before making any new trusted trial.')
        for attempt in trial.get('attempts', []):
            if attempt.get('error'):
                st.error(f'Qwen patch rejected: {attempt["error"]}')
        st.info('**Scientific integrity status: UNVERIFIABLE**, pending a researcher-reviewed paper contract and independent evaluator.')
        with st.expander('Code changes and execution errors', expanded=True):
            proposed = next((a.get('diff') for a in trial.get('attempts', [])
                             if a.get('requires_human_review')), None)
            st.code(proposed or trial.get('final_diff') or '(no source change)', language='diff')
            st.code(trial['final_execution']['stderr_tail'] or '(no captured error)', language='text')
        st.download_button('Download actual trial evidence', json.dumps(trial, indent=2),
                           file_name='research_trial.json', mime='application/json')
        st.caption('Candidate copy: ' + str(Path(st.session_state['science_trial_path']) / 'candidate_copy'))

    st.divider()
    st.markdown('### B. Scientific Integrity Gate (framework-independent)')
    st.write('A source-code repair is **not scientifically verified** just because it executes. '
             'This gate compares a reviewed paper-specific methodology contract, locked datasets/configuration, '
             'approved source changes, and measurements computed by an independent evaluator.')
    st.caption('Prepare a contract using `python -m scripts.make_science_contract --help`. '
               'Its original research commit, methodology and acceptance tolerance require researcher review.')
    science_backend_choice = st.radio('Independent evaluation environment',
                                       ['Local Python (trusted only)', 'Docker (restricted)'],
                                       horizontal=True, key='science_execution_backend')
    science_local = science_backend_choice.startswith('Local')
    with st.form('scientific_integrity_gate'):
        contract = st.text_input('Paper scientific contract JSON (local path)', placeholder='contracts/paper.json')
        original = st.text_input('Original repository source folder',
                                 value=st.session_state.get('science_trial_original', ''))
        candidate = st.text_input('Repaired source folder',
                                  value=(str(Path(st.session_state['science_trial_path']) / 'candidate_copy')
                                         if st.session_state.get('science_trial_path') else ''))
        evaluator = st.text_input('Independent evaluator Python file (outside candidate)',
                                  placeholder='trusted_evaluators/paper_eval.py')
        if science_local:
            verified_local_python = st.text_input('Dedicated trusted evaluator venv Python executable',
                                                 placeholder=r'C:\research_env\Scripts\python.exe')
            verified_image = ''
            st.caption('Your approved contract must contain local_environment_sha256 matching the venv. '
                       'Generate its fingerprint with: python -m resource_ai.local_runtime --python PATH_TO_VENV_PYTHON')
            evaluator_local_trust = st.checkbox('I reviewed and fully trust the evaluator and research code on my computer')
        else:
            verified_image = st.text_input('Pinned preinstalled Docker image (repo@sha256:...)',
                                           value='python:3.11-slim', key='science_docker_image')
            verified_local_python = ''
            evaluator_local_trust = False
        verify_button = st.form_submit_button('Verify scientific integrity')
    if verify_button:
        problems = check_science_inputs(contract, original, candidate, evaluator)
        if problems:
            for problem in problems:
                st.error(problem)
            st.info('Complete the original trial and obtain a researcher-reviewed paper contract before independent verification.')
        elif science_local and not evaluator_local_trust:
            st.error('Local evaluation requires confirmation that you trust the code.')
        else:
            try:
                with st.spinner('Checking locked methodology and running independent evaluation if possible...'):
                    report = verify_science(contract, original, candidate,
                                            evaluator_path=evaluator or None, docker_image=verified_image,
                                            execution_backend='local' if science_local else 'docker',
                                            local_python=verified_local_python if science_local else None,
                                            allow_trusted_local=science_local and evaluator_local_trust)
                output = unique_run_dir(RUNS, 'science_evidence')
                (output / 'SCIENTIFIC_REPORT.md').write_text(as_markdown(report), encoding='utf-8')
                (output / 'scientific_result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                st.session_state['science_report'] = report
            except Exception as exc:
                st.error(f'Scientific review could not complete: {type(exc).__name__}: {exc}')
    report = st.session_state.get('science_report')
    if report:
        verdict = report['verdict']
        if verdict == 'PASSED_SPECIFIED_SCIENTIFIC_CHECKS':
            st.success('Specified scientific checks passed. Review the evidence and scope limits below.')
        elif verdict == 'METHODOLOGY_VIOLATION':
            st.error('Protected methodology/data changed. Repair is not acceptable.')
        else:
            st.warning('Scientific verdict: ' + verdict)
        st.dataframe(pd.DataFrame(report['checks']), hide_index=True, use_container_width=True)
        stats = pd.DataFrame([{'status': key, 'count': sum(ch['status'] == key for ch in report['checks'])}
                              for key in sorted(set(ch['status'] for ch in report['checks']))]).set_index('status')
        st.bar_chart(stats)
        if report.get('measured_metric') is not None:
            st.metric('Independently measured metric', report['measured_metric'])
            ref = report.get('declared_reference_metric') or {}
            if isinstance(ref.get('value'), (float, int)):
                st.caption('This comparison is from an independent evaluator and the researcher-locked reference, '
                           'not candidate-printed accuracy.')
                vals = pd.DataFrame([{'run': 'Published/reference target', 'observed': ref['value']},
                                     {'run': 'Independent evaluator', 'observed': report['measured_metric']}]).set_index('run')
                st.bar_chart(vals)
        st.download_button('Download scientific evidence JSON', json.dumps(report, indent=2),
                           file_name='scientific_result.json', mime='application/json')
        st.download_button('Download scientific methodology report', as_markdown(report),
                           file_name='SCIENTIFIC_REPORT.md', mime='text/markdown')
        st.caption(report['limits'])


def validation_evidence():
    st.divider()
    st.markdown('### Controlled scientific-integrity validation on real datasets')
    report = ROOT / 'results' / 'official_validation_run.json'
    if not report.is_file():
        st.info('No validation results in this release. Run: python validation/real_data_pilot.py '
                '--out results/official_validation_run.json')
        return
    try:
        data = json.loads(report.read_text(encoding='utf-8'))
        rows = data.get('cases') or []
        if not rows:
            st.info('No completed validation cases recorded.')
            return
        summary = validation_summary(data)
        a,b,c = st.columns(3)
        a.metric('Observed checks', summary['total_cases'])
        b.metric('Correctly classified', summary['verifier_matches_expected'])
        c.metric('Invalid cases accepted by score-only', summary['invalid_cases_score_rule_accepted'])
        st.dataframe(pd.DataFrame([{
            'Real dataset': r['dataset'], 'Controlled modification': r['failure_type'],
            'Expected': r['expected'], 'Observed': r['actual'],
            'Correct classification': r['passed'], 'Score-only accepted': r['score_only_accept']
        } for r in rows]), hide_index=True, use_container_width=True)
        plot = pd.DataFrame([{'dataset': d,
            'passed_checks': sum(r.get('passed') is True for r in rows if r['dataset']==d),
            'failed_checks': sum(r.get('passed') is not True for r in rows if r['dataset']==d)}
            for d in summary['datasets']]).set_index('dataset')
        st.bar_chart(plot)
        reference_metrics = []
        for dataset in summary['datasets']:
            trace_file = ROOT / 'results' / 'real_data_references' / f'{dataset}.trace.json'
            if trace_file.exists():
                trace = json.loads(trace_file.read_text(encoding='utf-8'))
                row = {'dataset': dataset}
                for event in trace.get('events', []):
                    if event.get('type') == 'metric' and event.get('depth', 0) == 0 and isinstance(event.get('value'), (int, float)):
                        row[event.get('name', 'metric')] = event['value']
                if len(row) > 1:
                    reference_metrics.append(row)
        if reference_metrics:
            st.markdown('**Recorded reference-model metrics**')
            st.dataframe(pd.DataFrame(reference_metrics), hide_index=True, use_container_width=True)
            st.bar_chart(pd.DataFrame(reference_metrics).set_index('dataset'))
        st.caption('Actual controlled checks on real datasets; the code modifications were intentionally '
                   'constructed. This chart is not a published-paper recovery rate.')
        st.download_button('Download real-data evidence', report.read_bytes(),
                           file_name='official_validation_run.json', mime='application/json')
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Could not read saved validation evidence: {exc}')


def history():
    title('04 / Evidence', 'Report history', 'Previous local inspections and runs, kept for teacher review.')
    results = recorded_cases(RUNS)
    if results:
        st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
        choices = [x['path'] for x in results]
        path = Path(st.selectbox('Open saved record', choices))
        try:
            obj = json.loads(path.read_text(encoding='utf-8'))
            with st.expander('JSON evidence', expanded=True):
                st.json(obj, expanded=False)
            st.download_button('Download this record', path.read_bytes(), file_name=path.name,
                               mime='application/json')
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    else:
        st.info('No local reports yet. Start a repository scan or controlled demo.')
    validation_evidence()
    st.divider()
    st.markdown('### What this release can and cannot do')
    st.markdown('''
- **Can:** Inspect one GitHub link or a CSV/Markdown queue, record available source and possible compatibility concerns, save evidence files, and display real inspection statistics.
- **Can:** Run bundled, known benchmark examples through the existing scripted or optional Ollama repair workflow.
- **Cannot:** Confirm that a downloaded repository is broken from static inspection alone.
- **Cannot:** Safely execute unknown GitHub code on your desktop. Untrusted code requires appropriate isolation. Local execution is opt-in for fully trusted code only.
- **Cannot:** Independently reproduce arbitrary scientific papers or certify PyTorch/TensorFlow methodology in this alpha.

The next research milestone is a supervised, isolated recovery trial with a **known paper, exact repository commit, original dataset and reference methodology**.
''')


with st.sidebar:
    st.markdown('### 🧪 RE:SOURCE AI')
    st.caption('Research Recovery Console')
    st.caption('Official Research Pilot')
    st.divider()
    selected = st.radio('Workspace', ['Scan a Link', 'Research Queue', 'Research Trial', 'Controlled Recovery', 'Reports & About'],
                        label_visibility='collapsed')
    st.divider()
    st.caption('LOCAL RESEARCH TOOL')
    st.caption('GitHub source inspection does not execute third-party code.')
    st.caption('Science checks require a locked, researcher-reviewed contract and trusted evaluator.')

if selected == 'Scan a Link':
    single_repo()
elif selected == 'Research Queue':
    research_queue()
elif selected == 'Research Trial':
    research_trial()
elif selected == 'Controlled Recovery':
    controlled_demo()
else:
    history()

st.markdown(f'<div class="rs-footer">RE:SOURCE AI · '
            'Inspection evidence is separate from scientific verification · Open research prototype</div>',
            unsafe_allow_html=True)
