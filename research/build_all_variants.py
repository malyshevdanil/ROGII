"""Build several 160-seed submission variants, all with new stages placed AFTER gold-calibration
(the fix confirmed necessary by the wall-hedge real-LB regression)."""
import json, copy

ROOT = 'd:/ROGII'
BASE = f'{ROOT}/7.178_diversepf_moreseeds192.ipynb'

def load_base_160():
    nb = json.load(open(BASE, encoding='utf-8'))
    for c in nb['cells']:
        if c['cell_type'] != 'code': continue
        src = ''.join(c['source'])
        if 'PF_SEEDS = 192' in src:
            new_src = src.replace(
                'PF_SEEDS = 192  # max variance-reduction (160->192, ~+20% runtime, watch 9h)  # more-seeds: variance reduction to remove seed-luck (was 128)',
                "PF_SEEDS = 160  # SAFETY MARGIN: reduced from 192 to buy back runtime headroom for the new\n"
                "    # postprocessing stages added on top of this base -- validated within the pipeline's own\n"
                "    # seed-noise floor (~0.07) vs 192 seeds (7.091 vs 7.080)."
            )
            assert new_src != src
            c['source'] = [l + '\n' for l in new_src.rstrip('\n').split('\n')]
    return nb

def find_idx(cells, marker):
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code' and marker in ''.join(c['source']):
            return i
    return None

def get_cell_src(path, marker):
    nb = json.load(open(path, encoding='utf-8'))
    i = find_idx(nb['cells'], marker)
    assert i is not None, f'{marker} not found in {path}'
    return copy.deepcopy(nb['cells'][i])

WALLHEDGE_CELL = get_cell_src(f'{ROOT}/7.178_moreseeds192_wallhedge.ipynb', 'WALL HEDGE')
WARP_CELL = get_cell_src(f'{ROOT}/7.178_moreseeds192_full_stack.ipynb', 'WARP DECORRELATION')
PP_CELL = get_cell_src(f'{ROOT}/7.178_moreseeds192_full_stack.ipynb', 'PHYSICS POST-PROCESS')
PFZ_CELL = get_cell_src(f'{ROOT}/7.178_moreseeds192_full_stack.ipynb', 'PF_Z BLEND')

def build_variant(stage_cells, dst_name):
    nb = load_base_160()
    cells = nb['cells']
    final_i = find_idx(cells, '_final = strict_audit("FINAL"')
    assert final_i is not None
    for offset, cell in enumerate(stage_cells):
        cells.insert(final_i + offset, copy.deepcopy(cell))
    nb['cells'] = cells
    dst = f'{ROOT}/{dst_name}'
    json.dump(nb, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('wrote', dst, 'cells=', len(cells))
    return dst

build_variant([WALLHEDGE_CELL], '7.178_moreseeds160_wallhedge_aftergold.ipynb')
build_variant([WARP_CELL], '7.178_moreseeds160_warpblend_aftergold.ipynb')
build_variant([WARP_CELL, PP_CELL], '7.178_moreseeds160_warpblend_physicspp_aftergold.ipynb')
# full_stack (warp+pp+pfz) at 160, after-gold already exists as 7.178_moreseeds160_full_stack_aftergold.ipynb
print('done')
