import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
STYLES = {'nsga2': ('NSGA-II + ABC', '#697887', '--'), 'lcb': ('GUALO-LCB', '#008b90', '-')}

def curve(path, reference, seconds):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    points = []
    best = -math.inf
    for row in sorted(rows, key=lambda r: r['elapsed']):
        if not 0 <= row['elapsed'] <= seconds:
            continue
        q = 100 * (2 - row['area'] / reference[0] - row['delay'] / reference[1])
        if not math.isfinite(q):
            raise ValueError(f'Invalid QoR in {path}')
        if q > best:
            best = q
            points.append((row['elapsed'] / 3600, best))
    if points and points[-1][0] < seconds / 3600:
        points.append((seconds / 3600, best))
    return points

def plot_results(run, seconds=None, output=None):
    run = Path(run)
    protocol = json.loads((run / 'protocol.json').read_text())
    budget = float(protocol.get('seconds', protocol.get('seconds_per_arm', 3600)))
    seconds = budget if seconds is None else min(float(seconds), budget)
    if seconds <= 0:
        raise ValueError('Plot duration must be positive.')
    results = json.loads((run / 'results.json').read_text())
    methods = [m for m in STYLES if any((r['method'] == m for r in results))]
    designs = list(dict.fromkeys((r['design'] for r in results if r['method'] in methods)))
    if not designs:
        return []
    output = Path(output) if output is not None else run / 'qor_comparison'
    output.parent.mkdir(parents=True, exist_ok=True)
    panels = {}
    for design in designs:
        panels[design] = {}
        for method in methods:
            matches = [r for r in results if r['design'] == design and r['method'] == method]
            if not matches:
                continue
            record = matches[0]
            if record['status'] != 'complete':
                raise ValueError(f'Incomplete run: {design}/{method}')
            reference = record['reference']
            if not reference or not all((math.isfinite(v) and v > 0 for v in reference)):
                raise ValueError(f'Missing ABC reference: {design}/{method}')
            horizon = seconds
            live = run / 'live' / f'{design}_{method}.json'
            if live.is_file():
                state = json.loads(live.read_text())
                horizon = min(horizon, float(state['elapsed']))
            path = run / design / method / f'{design}_{method}' / 'abc.jsonl'
            points = curve(path, reference, horizon) if path.is_file() else []
            panels[design][method] = points
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#adbbc6', 'text.color': '#223645', 'axes.labelcolor': '#223645', 'xtick.color': '#405767', 'ytick.color': '#405767', 'savefig.facecolor': 'white'}):
        columns = min(2, len(designs))
        rows = math.ceil(len(designs) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(7.2 * columns, 4.4 * rows + 0.5), squeeze=False)
        hours = seconds / 3600
        for ax, design in zip(axes.flat, designs):
            endpoints = []
            values = [q for pts in panels[design].values() for _, q in pts]
            if values:
                low, high = (min(values), max(values))
                span = max(high - low, abs(high) * 0.03, 0.1)
                ax.set_ylim(low - 0.07 * span, high + 0.15 * span)
            else:
                low, high, span = (0.0, 1.0, 1.0)
                ax.text(0.5, 0.5, 'No ABC measurements', transform=ax.transAxes, ha='center')
            for method, pts in panels[design].items():
                if not pts:
                    continue
                label, color, style = STYLES[method]
                x, y = zip(*pts)
                ax.step(x, y, where='post', color=color, ls=style, lw=2.8)
                ax.scatter(x[-1], y[-1], s=46, color=color, edgecolors='white', linewidths=0.7, zorder=5)
                endpoints.append((y[-1], x[-1], color))
            endpoints.sort()
            positions = []
            for val, _, _ in endpoints:
                positions.append(max(val, positions[-1] + 0.085 * span) if positions else val)
            if positions and positions[-1] > high + 0.09 * span:
                delta = positions[-1] - high - 0.09 * span
                positions = [v - delta for v in positions]
            for (value, x, color), y in zip(endpoints, positions):
                ax.plot([x + 0.005 * hours, hours * 1.035], [value, y], color=color, lw=1)
                ax.text(hours * 1.055, y, f'{value:.3f}', color=color, va='center', fontsize=12, fontweight='bold')
            ax.set_xlim(0, hours * 1.24)
            ax.set_xticks([hours * i / 6 for i in range(7)])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.3g}'))
            ax.yaxis.set_major_locator(MaxNLocator(6))
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Best measured Q × 100')
            ax.set_title(design, fontsize=20, fontweight='bold', pad=12)
            ax.axvline(hours, color='#becad3', lw=1, ls=':')
            ax.grid(alpha=0.2)
            ax.set_axisbelow(True)
        for ax in list(axes.flat)[len(designs):]:
            ax.set_visible(False)
        handles = [Line2D([0], [0], color=STYLES[m][1], ls=STYLES[m][2], lw=3, label=STYLES[m][0]) for m in methods]
        fig.legend(handles=handles, loc='lower center', ncol=min(2, len(handles)), frameon=False, bbox_to_anchor=(0.5, 0.015))
        fig.tight_layout(rect=[0, 0.18 / rows, 1, 1], w_pad=3)
        paths = [output.with_suffix('.png'), output.with_suffix('.pdf')]
        for path in paths:
            fig.savefig(path, dpi=200)
        plt.close(fig)
    return paths
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    for path in plot_results(args.run, args.seconds, args.output):
        print(path)
