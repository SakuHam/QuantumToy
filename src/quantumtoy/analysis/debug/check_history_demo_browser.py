"""Check real browser controls and compare the embedded law to Python.

Start geckodriver on localhost:4444 with a profile root accessible to Firefox.
This checker uses only the standard WebDriver HTTP protocol, no JS packages.
"""
import argparse
import base64
import json
from pathlib import Path
import sys
import urllib.request
import urllib.error
from contextlib import contextmanager
import subprocess
import tempfile
import time
from urllib.parse import urlparse

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.temporal_history_profile import TemporalHistoryEnvelope, selection_time_weights
from analysis.memory_banks import MemoryBankConfig, evaluate_memory_banks

ROOT = Path(__file__).resolve().parents[4]


@contextmanager
def local_driver(args):
    if not args.start_driver:
        yield
        return
    endpoint = urlparse(args.webdriver_url)
    with tempfile.TemporaryDirectory(prefix='browser-check-', dir=ROOT) as profile:
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(['geckodriver', '--host', endpoint.hostname,
                '--port', str(endpoint.port), '--profile-root', profile], stdout=log, stderr=log)
            try:
                for _ in range(100):
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read().decode())
                    try:
                        with urllib.request.urlopen(args.webdriver_url+'/status', timeout=1):
                            break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(.1)
                else:
                    raise RuntimeError('Local WebDriver did not start')
                yield
            finally:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                except PermissionError:
                    # Some Snap/AppArmor installations prohibit signalling the
                    # driver even though WebDriver DELETE closed the browser.
                    print(f'Browser closed; environment denied driver cleanup (PID {process.pid}).', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--webdriver-url', default='http://127.0.0.1:4444')
    parser.add_argument('--start-driver', action='store_true', help='Start a local geckodriver; clean it up when permitted')
    args = parser.parse_args()
    with local_driver(args):
        check(args)


def check(args):
    def request(path, payload=None, method=None):
        req = urllib.request.Request(args.webdriver_url+path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.load(response)['value']
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from error

    session = request('/session', {'capabilities': {'alwaysMatch': {
        'browserName': 'firefox', 'moz:firefoxOptions': {'args': ['-headless']}}}})['sessionId']
    prefix = '/session/'+session

    def js(script, *args):
        # Firefox's WebDriver sandbox cannot directly see page-global lexical
        # declarations. Evaluate this fixed, local checker code in the page.
        code = '(function(){'+script+'}).apply(null,'+json.dumps(list(args))+')'
        return request(prefix+'/execute/sync', {'script': 'return window.eval(arguments[0])', 'args': [code]})

    try:
        request(prefix+'/window/rect', {'width': 1440, 'height': 1200})
        request(prefix+'/url', {'url': (ROOT/'demo/trf_clock_lab.html').as_uri()})
        assert js("return document.documentElement.lang") == 'en'
        js("document.querySelector('[data-lang=fi]').click()")
        assert js("return document.documentElement.lang") == 'fi'
        assert 'Aaltofunktio' in js("return document.getElementById('dynamicsPanel').textContent")
        js("document.querySelector('[data-lang=en]').click(); state.playing=false;")
        data = json.loads((ROOT/'history_dynamics_study.json').read_text())['demo']
        basis = np.asarray(data['clicks'])
        densities = np.asarray(data['densities'])
        blur = np.asarray(data['blur'])
        cases = [(.05, 1, .5, .5, 0), (.1, 2, 5, 1.5, 3),
                 (.5, 0, 1, 2, 0), (1, 1, 5, 1.5, 0),
                 (2, 2, 12, 3, 20)]
        for g, strength, fade, beta, wait in cases:
            result = js("""
                document.querySelector(`[data-g='${arguments[0]}']`).click();
                for(const [id,value] of [['lambda',arguments[1]],['historyFade',arguments[2]],
                    ['historyBeta',arguments[3]],['readoutWait',arguments[4]]]) {
                    const el=document.getElementById(id);el.value=value;el.dispatchEvent(new Event('input'));
                }
                return {law:lastDynamics,sigma:current().sig,alpha:current().alpha};
            """, g, strength, fade, beta, wait)
            sigma = result['sigma']
            env = TemporalHistoryEnvelope(sigma_t=sigma, retention_time=1.5*sigma,
                                           fade_time=fade*sigma, fade_power=beta)
            weights = selection_time_weights(data['edges'], env,
                front_time=-2+sigma/result['alpha'], strength=strength)
            ideal = np.einsum('c,cty->ty', weights, basis)
            density = weights@densities
            genuine = data['efficiency'] * (ideal@blur.T)
            dark = (1-genuine.sum())*data['darkProbability']/genuine.size
            age = data['duration']+wait*sigma-np.asarray(data['times'])
            survival = np.exp(-np.maximum((age-env.retention_time)/env.fade_time, 0)**beta)
            saved = (genuine+dark)*survival[:, None]
            actual = result['law']
            assert_allclose(actual['weights'], weights, atol=2e-6, rtol=2e-5)
            assert_allclose(actual['ideal'], ideal, atol=2e-8, rtol=2e-5)
            assert_allclose(actual['density'], density, atol=2e-7, rtol=2e-5)
            assert_allclose(actual['saved'], saved, atol=2e-8, rtol=2e-5)
            assert abs(actual['savedMass']+actual['erased']+actual['missing']-1) < 1e-12
            print(f'Browser/Python agreement: g={g}, lambda={strength}, fade={fade}, beta={beta}, wait={wait}')

        # The same chosen branch mixture must survive changes of readout memory.
        js("""
            document.querySelector('[data-g="1"]').click();
            for(const [id,value] of [['lambda',1],['historyFade',5],
                ['historyBeta',1.5],['readoutWait',0]]) {
                const el=document.getElementById(id);el.value=value;el.dispatchEvent(new Event('input'));
            }
        """)
        before = js('return lastDynamics')
        js("document.getElementById('historyFade').value=.5; document.getElementById('historyFade').dispatchEvent(new Event('input'));")
        after = js('return lastDynamics')
        assert_allclose(before['density'], after['density'], atol=0, rtol=0)
        assert after['savedMass'] < before['savedMass']
        js("state.historyFade=5;document.getElementById('historyFade').value=5;update();")
        before_memory = js('return lastDynamics')
        memory_cases = [
            (0, 0, 1, 1, 1, 'independent'),
            (2, 0, 4, 1, .9, 'independent'),
            (0, 1, 1, 1, 1, 'independent'),
            (0, 1, 8, 1, 1, 'independent'),
            (1, 4, 4, 1, .9, 'independent'),
            (1, 4, 4, 1, .9, 'shared'),
            (0, 16, 16, 0, .4, 'shared'),
            (0, 16, 16, 5, 0, 'independent'),
        ]
        for refs, copies, reads, spacing, efficiency, mode in memory_cases:
            result = js("""
                for(const [id,value] of Object.entries(arguments[0])) {
                    const el=document.getElementById(id);el.value=value;el.dispatchEvent(new Event('input'));
                }
                return {memory:lastMemoryBanks,physics:lastDynamics,sigma:current().sig};
            """, dict(referenceCopies=refs, delayedCopies=copies, memoryReads=reads,
                       memorySpacing=spacing, memoryReadEfficiency=efficiency, memoryLossMode=mode))
            sigma = result['sigma']
            config = MemoryBankConfig(refs, copies, .995, mode, reads, spacing*sigma, efficiency)
            envelope = TemporalHistoryEnvelope(sigma_t=sigma, retention_time=1.5*sigma,
                                               fade_time=5*sigma, fade_power=1.5)
            expected = evaluate_memory_banks(before_memory['before'], data['times'], envelope,
                                            readout_time=data['duration'], config=config)
            for js_key, py_key in [('reference','reference_joint'), ('delayed','delayed_joint'),
                                   ('anyRead','any_read_joint'), ('either','either_joint'), ('both','both_joint')]:
                assert_allclose(result['memory'][js_key], getattr(expected, py_key), atol=1e-13)
            assert_allclose(result['memory']['probabilities'], expected.probabilities, atol=1e-13)
            assert_allclose(result['physics']['ideal'], before_memory['ideal'], rtol=0, atol=0)
            assert_allclose(result['physics']['density'], before_memory['density'], rtol=0, atol=0)
            print(f'Memory banks/Python agreement: refs={refs}, copies={copies}, reads={reads}, mode={mode}')
        js("""
            for(const [id,value] of Object.entries({referenceCopies:1,delayedCopies:4,
                memoryReads:4,memorySpacing:1,memoryReadEfficiency:.9,memoryLossMode:'independent'})) {
                const el=document.getElementById(id);el.value=value;el.dispatchEvent(new Event('input'));
            }
            document.querySelector('[data-lang=fi]').click();
        """)
        assert 'Vertailumuistit' in js("return document.getElementById('memoryPanel').textContent")
        # Neither language may require viewers to have the private papers.
        assert 'TRF-IT' not in js('return document.body.innerText')
        request(prefix+'/window/rect', {'width': 420, 'height': 1000})
        assert js('return document.documentElement.scrollWidth <= innerWidth+1'), 'Mobile horizontal overflow'
        js("document.querySelector('[data-lang=en]').click()")
        request(prefix+'/window/rect', {'width': 1440, 'height': 1200})
        element = request(prefix+'/element', {'using': 'css selector', 'value': '#dynamicsPanel'})
        element_id = next(iter(element.values()))
        png = request(prefix+'/element/'+element_id+'/screenshot')
        (ROOT/'demo/history_dynamics_preview.png').write_bytes(base64.b64decode(png))
        element = request(prefix+'/element', {'using': 'css selector', 'value': '#memoryPanel'})
        element_id = next(iter(element.values()))
        png = request(prefix+'/element/'+element_id+'/screenshot')
        (ROOT/'demo/memory_banks_preview.png').write_bytes(base64.b64decode(png))
        print('Language switch, slider events, erasure invariance, mobile width and screenshot: OK')
    finally:
        request(prefix, method='DELETE')


if __name__ == '__main__':
    main()
