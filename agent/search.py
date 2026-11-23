import traceback
from common import PoolMixin, BudgetEnded, RESYN2, save, costs
from nsga2 import SearchBase, PilotMixin, fast_selection

class GUALOSearch(PoolMixin, PilotMixin, SearchBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa = float(self.config.get('kappa', 1.0))

    def select(self, pool, predictions, count):
        scores = costs(predictions, self.reference, self.kappa)
        indices = fast_selection(scores, count)
        return ([pool[i] for i in indices], scores)

    def run(self):
        n = self.config['population']
        try:
            self.publish('baseline')
            self.reference = self.abc(self.design, None, RESYN2, min(180.0, self.remaining()))
            self.publish('initial')
            initial = self.fresh(self.config['initial_pool'])
            predictions = self.predict(initial)
            chosen, scores = self.select(initial, predictions, n)
            self.selected = chosen
            self.seen.difference_update(set(initial) - set(chosen))
            save(self.out / 'selection_000000.json', dict(pool=[list(s) for s in initial], scores=scores, selected=[list(s) for s in chosen], unknown=[list(s) for s in chosen]))
            self.label(chosen)
            self.pop = list(chosen)
            self.record()
            while self.remaining() > 0 and (not self.config.get('max_generations') or self.generation < self.config['max_generations']):
                self.generation += 1
                self.publish('variation')
                parents = list(self.pop)
                children = self.evolve()
                injection_count = round(self.config['alpha'] * n)
                fresh = self.fresh(self.config['injection_pool']) if injection_count else []
                injected, _ = self.select(fresh, self.predict(fresh), injection_count) if fresh else ([], [])
                self.seen.difference_update(set(fresh) - set(injected))
                merged = list(dict.fromkeys(parents + children + injected))
                predictions = self.predict(merged)
                chosen, scores = self.select(merged, predictions, n)
                unknown = [s for s in chosen if s not in self.truth]
                self.counts = dict(parents=len(parents), offspring=len(children), injected=len(injected), merged=len(merged), selected=n, new_abc=len(unknown))
                self.selected = list(chosen)
                self.candidates = [dict(seq=list(s), selected=s in chosen, area_cost=scores[i][0], delay_cost=scores[i][1]) for i, s in enumerate(merged)]
                save(self.out / f'selection_{self.generation:06d}.json', dict(parents=[list(s) for s in parents], offspring=[list(s) for s in children], injected=[list(s) for s in injected], pool=[list(s) for s in merged], scores=scores, selected=[list(s) for s in chosen], unknown=[list(s) for s in unknown]))
                self.publish('select')
                self.label(unknown)
                self.pop = list(chosen)
                self.seen.difference_update(set(merged) - set(chosen) - set(self.truth))
                self.record()
            self.status = 'complete'
        except BudgetEnded:
            self.status = 'complete'
        except Exception as exc:
            self.status = 'failed'
            self.error = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
        finally:
            self.workers.shutdown(wait=True)
            if self.reference and self.truth and (not self.history or len(self.truth) > self.history[-1]['abc_succeeded']):
                self.record(complete=False)
            self.publish('finished' if self.status == 'complete' else 'failed')
