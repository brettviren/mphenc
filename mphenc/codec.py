#!/usr/bin/env python3
'''
Median progressive Huffman codec
'''

from math import ceil
import numpy
import huffman
from bitarray import bitarray

def entropy(img):
    '''
    Return the entropy of the image in units of bits
    '''
    codes, count = numpy.unique(img, return_counts=True)
    norm = numpy.sum(count)
    cnorm = count/norm
    return -numpy.sum(cnorm * numpy.log2(cnorm))

def huffcodes(img):
    '''
    Return Huffman code table for image
    '''
    codes, counts = numpy.unique(img, return_counts=True)
    cb = huffman.codebook(zip(codes,counts))
    return {k:bitarray(v) for k,v in cb.items()}

def huffenc(img, codebook):
    '''
    Return Huffman encoding for img.
    '''
    img = img.reshape(-1)
    ret = bitarray()
    for code in img:
        ret += bitarray(codebook[code])
    return ret.tobytes()

def huffencodes(img):
    '''
    Return (enc,codes) for image
    '''
    cb = huffcodes(img)
    bits = huffenc(img, cb)
    return bits, cb

def sizecmp(img, codesize=12, include_codebook=False):
    bits,cb = huffencodes(img)
    native = codesize * img.shape[0] * img.shape[1]
    compressed = len(bits)
    if include_codebook:
        compressed += codesize*len(cb)
        for b in cb.values():
            compressed += len(b)
    return native, compressed

def chunkup(img, size):
    '''
    Break up img into two: (med,sub).
    '''
    nchunks = int(img.shape[1]/size)
    meds = list()
    subs = list()
    for ind in range(nchunks):
        chunk = img[:, ind*size:(ind+1)*size]
        med = chunk[0]
        sub = chunk[:, 1:] - chunk[:, :-1]
        # med = numpy.array(numpy.median(chunk, axis=1), dtype=int)
        # sub = (chunk.T-med).T
        meds.append(med)
        subs.append(sub)
    meds = numpy.vstack(meds).T
    subs = numpy.hstack(subs)
    return meds,subs

def test(img, nslices = 100, include_codebook=False):
    meds = list()
    slicelen = int(round(img.shape[1] / nslices))
    totsize = 0;
    for n in range(nslices):
        chunk = img[:, n*slicelen:(n+1)*slicelen]
        med = numpy.array(numpy.median(chunk, axis=1), dtype=int)
        normed = (chunk.T - med).T
        meds.append(med)
        ent = entropy(normed)
        nat, siz = sizecmp(normed, include_codebook=include_codebook)
        totsize += siz
        print(f'\t{n}: {normed.shape} x {ent} = {siz}')

    meds = numpy.asarray(meds)
    ment = entropy(meds)
    nat, msiz = sizecmp(meds, include_codebook=include_codebook)
    print(f'medians: {meds.shape} x {ment} = {msiz}')
    csize = msiz + totsize

    ient = entropy(img)
    nat, isize = sizecmp(img, include_codebook=include_codebook)
    print(f'input: {img.shape} x {ient} = {isize}')
    
    native = 12 * img.shape[0] * img.shape[1]

    print(f'compressed: {csize}, factor: {isize/csize}')
    print(f'native: {native}, factor: {native/csize}')
    print(f'huffman: {native}, factor: {native/isize}')

def test2(img, size=10, include_codebook=False):
    meds, subs = chunkup(img, size)

    na,ca = sizecmp(img, 12, include_codebook)
    nm,cm = sizecmp(meds, 12, include_codebook)
    ns,cs = sizecmp(subs, 12, include_codebook)

    print("special:",(nm+ns)/(cm+cs))
    print("nominal:", na/ca)

### it actually inflates.  oh well....




