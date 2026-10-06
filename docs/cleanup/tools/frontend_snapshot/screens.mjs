import fs from 'fs';
import { withoutRetired } from './retiredDiseases.mjs';
import { summariseWhatIf } from './whatIfSummary.mjs';
import { responseError } from './apiError.mjs';
const isRetiredItem = (item) => item?.retired === true;           // ReviewQueuePanel / AdminDashboard rule
const fake = (r) => ({ ok: r.status < 400, status: r.status, statusText: r.statusText ?? '', json: async () => r.body });
const report = {};
for (const tag of ['old', 'new']) {
  const p = JSON.parse(fs.readFileSync(process.argv[2].replace('TAG', tag)));
  const r = {};
  r.picker = withoutRetired(p.diseases.body.diseases.map(d => ({ name: d.name, ...d.info }))).map(d => d.name);
  r.api = Object.fromEntries(Object.entries(p).filter(([k]) => /^(schema|predict|explain)\//.test(k)).map(([k, v]) => [k, v.status]));
  r.whatIf = Object.fromEntries(Object.entries(p).filter(([k]) => k.startsWith('cf/')).map(([k, v]) => {
    const s = summariseWhatIf({ counterfactuals: v.body.counterfactuals, prediction: v.prediction, baselineProbability: v.body.baseline_probability,
      patientData: v.patient, bestAchievable: v.body.best_achievable ?? null, message: v.body.message ?? null });
    return [k.slice(3), `${v.status} ${s.state}`];
  }));
  const items = p.queue.body.items, st = p.stats.body;
  const pendingListed = items.length;
  r.alQueue = {
    items_listed: pendingListed,
    read_only_cards: items.filter(isRetiredItem).map(i => i.disease),
    labellable_cards_by_disease: items.filter(i => !isRetiredItem(i)).reduce((a, i) => ({ ...a, [i.disease]: (a[i.disease] || 0) + 1 }), {}),
    stats_cards: ['Pending', ...(typeof st.retired_pending === 'number' ? ['Archived'] : []), 'Reviewed', 'Skipped'].join(' / '),
    pending_card: st.pending, archived_card: st.retired_pending ?? '(not shown)',
    pending_plus_archived_equals_listed: (st.pending + (st.retired_pending ?? 0)) === pendingListed,
  };
  const ann = p['annotate/diabetes'];
  r.annotateRetiredItem = ann ? `${ann.status} -> ${ann.status < 400 ? 'accepted (no message)' : 'shown: "' + await responseError(fake(ann)) + '"'}` : 'no diabetes item';
  r.skipHeartItem = p['skip/heart']?.status;
  r.history = Object.fromEntries(Object.entries(p).filter(([k]) => k.startsWith('history/')).map(([k, v]) => {
    const it = v.body.items || [];
    return [k.slice(8, 16), `${v.status}: ${it.length} rows (${[...new Set(it.map(i => i.disease))].join(',')}), archived note shown on ${it.filter(i => i.archived_note).length}`];
  }));
  r.adminRetrainHeart = p['retrain/heart'].status;
  report[tag === 'old' ? 'production backend (4c916e6)' : 'chore/retire-brfss backend'] = r;
}
console.log(JSON.stringify(report, null, 1));
