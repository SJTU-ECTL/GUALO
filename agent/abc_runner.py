import math, re, subprocess, tempfile
from common import BudgetEnded, OPS

class ABCOracle:

    def __init__(self, executable, liberty, aig_dir):
        self.executable = executable
        self.liberty = liberty
        self.aig_dir = aig_dir

    def __call__(self, name, seq, script, timeout):
        if timeout <= 0:
            raise BudgetEnded()
        flow = script or '; '.join((OPS[t] for t in seq))
        aig = self.aig_dir / (name + '.aig')
        for path in (aig, self.liberty):
            if any((c in str(path) for c in ('"', ';', chr(10), chr(13)))):
                raise ValueError('Unsupported ABC path character')
        command = f'read_aiger "{aig}"; read_lib "{self.liberty}"; strash; {flow}; map; print_stats'
        with tempfile.TemporaryDirectory(prefix='gualo_abc_') as directory:
            result = subprocess.run([str(self.executable), '-c', command], cwd=directory, capture_output=True, text=True, timeout=timeout)
        found = re.findall('area\\s*=\\s*([\\d.]+).*?delay\\s*=\\s*([\\d.]+)', result.stdout)
        if result.returncode or not found:
            raise RuntimeError(result.stdout[-500:] + result.stderr[-300:])
        values = list(map(float, found[-1]))
        if not all((math.isfinite(v) and v > 0 for v in values)):
            raise ValueError('Invalid ABC QoR')
        return values
