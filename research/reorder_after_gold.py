import json, sys

ROOT = 'd:/ROGII'
SRC = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/7.178_moreseeds160_full_stack.ipynb'
DST = sys.argv[2] if len(sys.argv) > 2 else f'{ROOT}/7.178_moreseeds160_full_stack_aftergold.ipynb'

nb = json.load(open(SRC, encoding='utf-8'))
cells = nb['cells']

def find_idx(marker):
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code' and marker in ''.join(c['source']):
            return i
    return None

warp_i = find_idx('WARP DECORRELATION')
pp_i = find_idx('PHYSICS POST-PROCESS')
pfz_i = find_idx('PF_Z BLEND')
gold_i = find_idx('def _run_gold_calibration')
final_i = find_idx('_final = strict_audit("FINAL"')

assert None not in (warp_i, pp_i, pfz_i, gold_i, final_i), (warp_i, pp_i, pfz_i, gold_i, final_i)
assert warp_i < pp_i < pfz_i < gold_i < final_i, "unexpected pre-existing order"

# pull the 3 cells out (in the order they currently appear: warp, pp, pfz)
moved = [cells[warp_i], cells[pp_i], cells[pfz_i]]

# remove them from the list, highest index first so lower indices stay valid
for i in sorted([warp_i, pp_i, pfz_i], reverse=True):
    del cells[i]

# recompute final_i after deletions (3 cells removed, all were before gold_i < final_i,
# so final_i shifts down by 3; recompute by search to be safe rather than trust arithmetic)
final_i2 = find_idx('_final = strict_audit("FINAL"')
assert final_i2 is not None

# insert the 3 moved cells (in original relative order) right before the final audit cell
for offset, cell in enumerate(moved):
    cells.insert(final_i2 + offset, cell)

nb['cells'] = cells
json.dump(nb, open(DST, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', DST, 'cells=', len(cells))

# sanity re-check final order
warp_i2 = find_idx('WARP DECORRELATION')
pp_i2 = find_idx('PHYSICS POST-PROCESS')
pfz_i2 = find_idx('PF_Z BLEND')
gold_i2 = find_idx('def _run_gold_calibration')
final_i3 = find_idx('_final = strict_audit("FINAL"')
print('new order: gold=%d < warp=%d < pp=%d < pfz=%d < final=%d ->' % (gold_i2, warp_i2, pp_i2, pfz_i2, final_i3),
      gold_i2 < warp_i2 < pp_i2 < pfz_i2 < final_i3)
