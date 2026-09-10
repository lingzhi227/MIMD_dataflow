"""Exact four-point FFT factorization and layout witness for the survey.

This independent mathematical checker does not execute SPIRAL or generate CSL.
Matrix coefficients are Gaussian integers, represented by pairs of Python ints.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Gaussian:
    re: int = 0
    im: int = 0

    def __add__(self, other):
        return Gaussian(self.re+other.re, self.im+other.im)

    def __mul__(self, other):
        return Gaussian(self.re*other.re-self.im*other.im,
                        self.re*other.im+self.im*other.re)

    def pair(self):
        return [self.re, self.im]


ZERO, ONE, NEG_I = Gaussian(), Gaussian(1), Gaussian(0, -1)


def matmul(a, b):
    if not a or not b or len(a[0]) != len(b):
        raise ValueError('Incompatible matrices')
    return [[sum((a[i][k]*b[k][j] for k in range(len(b))), ZERO)
             for j in range(len(b[0]))] for i in range(len(a))]


def kron(a, b):
    return [[av*bv for av in ar for bv in br] for ar in a for br in b]



def execute_layout(initial_owners, gathers, initial_transfers, exchange, output_layout):
    """Symbolically execute this fixed two-PE realization on all basis inputs."""
    memory = {0: {}, 1: {}}
    for index, owner in enumerate(initial_owners):
        memory[owner][f'x{index}'] = [ONE if j == index else ZERO for j in range(4)]

    def read(pe, name):
        if name not in memory[pe]:
            raise ValueError(f'{name} is unavailable on PE{pe}')
        return memory[pe][name]

    def move(transfer):
        src, dst, name = transfer['from'], transfer['to'], transfer['value']
        value = read(src, name)
        if src == dst:
            raise ValueError('A declared inter-PE transfer has identical endpoints')
        memory[dst][name] = value
        del memory[src][name]

    def combine(a, b, sign=1):
        return [x + Gaussian(sign)*y for x, y in zip(a, b)]

    for transfer in initial_transfers:
        move(transfer)
    for pe, (i, j) in enumerate(gathers):
        a, b = read(pe, f'x{i}'), read(pe, f'x{j}')
        prefix = 'e' if pe == 0 else 'o'
        memory[pe][prefix+'0'] = combine(a, b)
        memory[pe][prefix+'1'] = combine(a, b, -1)
    memory[1]['o1'] = [NEG_I*v for v in read(1, 'o1')]
    for transfer in exchange:
        move(transfer)
    memory[0]['y0'] = combine(read(0,'e0'), read(0,'o0'))
    memory[0]['y2'] = combine(read(0,'e0'), read(0,'o0'), -1)
    memory[1]['y1'] = combine(read(1,'e1'), read(1,'o1'))
    memory[1]['y3'] = combine(read(1,'e1'), read(1,'o1'), -1)
    result = {}
    for pe, names in output_layout.items():
        for name in names:
            if name in result:
                raise ValueError('Duplicate output owner')
            result[name] = read(int(pe), name)
    if set(result) != {'y0','y1','y2','y3'}:
        raise ValueError('Incomplete output layout')
    return [result[f'y{i}'] for i in range(4)]


def run():
    identity = [[ONE, ZERO], [ZERO, ONE]]
    butterfly = [[ONE, ONE], [ONE, Gaussian(-1)]]
    p = [0, 2, 1, 3]  # (P*x)[i] = x[p[i]]; gather convention.
    perm = [[ONE if j == p[i] else ZERO for j in range(4)] for i in range(4)]
    diagonal = [ONE, ONE, ONE, NEG_I]
    twiddle = [[diagonal[i] if i == j else ZERO for j in range(4)] for i in range(4)]
    roots = [ONE, NEG_I, Gaussian(-1), Gaussian(0, 1)]
    dft = [[roots[(i*j) % 4] for j in range(4)] for i in range(4)]
    first = kron(identity, butterfly)
    factorized = matmul(matmul(matmul(kron(butterfly, identity), twiddle), first), perm)
    if factorized != dft:
        raise AssertionError('FFT identity failed')
    # Verify the permutation-absorption identity on each row of both gathers.
    gathers = [[2*j+i for i in range(2)] for j in range(2)]
    composed = [[p[k] for k in gather] for gather in gathers]
    for gather, composition in zip(gathers, composed):
        g = [[ONE if j == k else ZERO for j in range(4)] for k in gather]
        composed_g = [[ONE if j == k else ZERO for j in range(4)] for k in composition]
        if matmul(g, perm) != composed_g:
            raise AssertionError('Individual gather-row composition failed')
    # Fixed block ownership x0,x1 on PE0; x2,x3 on PE1.
    initial_transfers = [{'value': f'x{k}', 'from': k//2, 'to': pe}
                         for pe, row in enumerate(composed) for k in row if k//2 != pe]
    # First-stage outputs e0,e1 on PE0 and o0,o1 on PE1; next-stage PE0 computes
    # y0,y2 while PE1 computes y1,y3. Twiddles are applied locally on PE1.
    exchange = [{'value': 'e1', 'from': 0, 'to': 1}, {'value': 'o0', 'from': 1, 'to': 0}]
    if len(initial_transfers) != 2:
        raise AssertionError('Unexpected initial redistribution')
    output_layout = {0: ['y0','y2'], 1: ['y1','y3']}
    owners = [0,0,1,1]
    spatial = execute_layout(owners, composed, initial_transfers, exchange, output_layout)
    if spatial != dft:
        raise AssertionError('Distributed symbolic execution disagrees with DFT definition')
    negatives = {}
    mutations = {'omitted_exchange': exchange[1:],
                 'misrouted_exchange': [dict(exchange[0], to=0), exchange[1]]}
    for name, mutated in mutations.items():
        try:
            execute_layout(owners, composed, initial_transfers, mutated, output_layout)
        except ValueError as error:
            negatives[name] = str(error)
        else:
            raise AssertionError('Invalid layout unexpectedly executed: '+name)
    return {'schema': 'spiral-four-point-witness.v1', 'status': 'PASS',
            'arithmetic': 'exact Gaussian integer coefficient matrices',
            'convention': 'F4[j,k]=exp(-2*pi*i*j*k/4), no normalization',
            'identity': 'F4=(F2 kron I2) diag(1,1,1,-i) (I2 kron F2) P',
            'matrix_entries_checked': 16, 'matrix': [[v.pair() for v in row] for row in factorized],
            'gather_permutation': p, 'composed_first_stage_gathers': composed,
            'initial_block_layout_transfers': initial_transfers,
            'first_to_second_stage_transfers': exchange,
            'initial_owners': owners, 'output_layout': output_layout,
            'spatial_execution_coefficients_checked': 16, 'negative_layout_results': negatives,
            'limitations': ['Two-PE exchange is a proposed fixed-layout realization, not a SPIRAL execution result',
                'Counts concern this scalar butterfly interface; no universal communication lower bound is claimed',
                'No timing, bank-conflict, DMA or finite-buffer feasibility is established by the matrix identity',
                'SPIRAL already includes distributed gather/send representations; this witness is not a novelty claim'],
            'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('PASS: 16 exact FFT coefficients; gather composition; fixed two-PE layout witness')
