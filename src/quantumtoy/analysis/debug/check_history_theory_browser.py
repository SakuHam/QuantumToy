"""Exercise the offline Theory viewer with a real Firefox browser."""
import argparse
import base64
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_history_demo_browser import local_driver

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--webdriver-url', default='http://127.0.0.1:4450')
    parser.add_argument('--start-driver', action='store_true')
    args = parser.parse_args()

    def request(path, payload=None, method=None):
        req = urllib.request.Request(args.webdriver_url+path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'}, method=method)
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)['value']

    with local_driver(args):
        session = request('/session', {'capabilities': {'alwaysMatch': {
            'browserName': 'firefox', 'moz:firefoxOptions': {'args': ['-headless']}}}})['sessionId']
        prefix = '/session/'+session

        def js(script, *values):
            code = '(function(){'+script+'}).apply(null,'+json.dumps(values)+')'
            return request(prefix+'/execute/sync', {'script': 'return window.eval(arguments[0])', 'args': [code]})

        try:
            request(prefix+'/window/rect', {'width': 1440, 'height': 1600})
            request(prefix+'/url', {'url': (ROOT/'demo/history_instrument_theory.html').as_uri()})
            report = json.loads((ROOT/'demo/history_instrument_theory.json').read_text())
            assert js('return DATA.runs.length') == len(report)
            for index, run in enumerate(report):
                final_time = run['reads'][-1]['time']
                values = js("""
                    $('scenario').value=arguments[0];$('scenario').dispatchEvent(new Event('change'));
                    $('timeline').value=arguments[1];$('timeline').dispatchEvent(new Event('input'));
                    return {detected:$('detected').textContent,reference:$('reference').textContent,
                        logged:$('logged').textContent,recovered:$('recovered').textContent,
                        fieldTime:run().times[frameIndex(run())],playing};
                """, index, final_time)
                assert values['detected'] == f"{100*run['diagnostics'][-1]['detected']:.1f} %"
                for key in ['reference', 'logged', 'recovered']:
                    assert int(values[key]) == run['reads'][-1][key]
                assert abs(values['fieldTime']-run['parameters']['dt']*run['parameters']['steps']) < 1e-9
                assert not values['playing']
                print(f'Theory viewer agrees with saved simulation: λ={run["parameters"]["selection_strength"]}')
            js("$('timeline').value=.8;$('timeline').dispatchEvent(new Event('input'));$('example').value=2;$('example').dispatchEvent(new Event('change'));$('fi').click()")
            assert js('return document.documentElement.lang') == 'fi'
            assert js("return $('logged').textContent") == '—'
            assert 'Valmistus '+str(js('return run().parameters.example_indices[2]')) in js("return $('exampleState').textContent")
            request(prefix+'/window/rect', {'width': 420, 'height': 1000})
            assert js('return document.documentElement.scrollWidth <= innerWidth+1')
            js("$('en').click();$('play').click()")
            request(prefix+'/execute/async', {'script': 'setTimeout(arguments[arguments.length-1],150)', 'args': []})
            assert js('return time') > .8
            js("$('play').click();$('reset').click()")
            assert js('return time') == 0
            assert not js('return playing')
            request(prefix+'/window/rect', {'width': 1440, 'height': 2800})
            js("$('scenario').value=1;$('scenario').dispatchEvent(new Event('change'));$('timeline').value=1.2;$('timeline').dispatchEvent(new Event('input'));window.scrollTo(0,0)")
            element = request(prefix+'/element', {'using': 'css selector', 'value': 'main'})
            png = request(prefix+'/element/'+next(iter(element.values()))+'/screenshot')
            (ROOT/'demo/history_instrument_preview.png').write_bytes(base64.b64decode(png))
            print('Timeline, scenario menu, individual trajectories, play/pause, EN/FI, mobile and screenshot: OK')
        finally:
            request(prefix, method='DELETE')


if __name__ == '__main__':
    main()
