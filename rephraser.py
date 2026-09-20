"""
Fork from https://github.com/travco/rephraser
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from signal import SIGINT, signal

import keyvi.compiler  # type: ignore[import-untyped]
import keyvi.dictionary  # type: ignore[import-untyped]
import markovify  # type: ignore[import-untyped]

BEGIN = '___BEGIN__'
END = '___END__'
DONE = '___DONE__'
undesirable_chars = [',','.',';',':','?','\'','"','"','"','"']

DCT = None  # Global mappings for shared memory managed by keyvi
DCT_KEYS: set[str] = set()  # Global set of all dictionary keys for iteration
mpqueue = None # Work queue
MAXQUEUESIZE = 100000  # Number of work items reasonable to have on queue
worker_num = 0  # Will be changed before creating workers

if sys.platform == "darwin":
  MAXQUEUESIZE=32767 #max allowed mp queue size on mac
  mp.set_start_method("fork") #necessary for workers to inherit global vars on mac
else:
  try:
    mp.set_start_method("fork")
  except RuntimeError:
    pass  # fork is not available on this platform

def sigint_handler(signal_received, frame):
  # Parent *should* be able to exit
  sys.stderr.write('[REPHRASER] SIGINT or CTRL-C detected. Attempting to exit gracefully...\n')
  last_work_item = mpqueue.get(block=False)
  sys.stderr.write('[REPHRASER] Next prefix in queue was: ' + repr(last_work_item)[2] + '\n')
  sys.exit(0)

def get_version() -> str:
    """Read the installed package version from distribution metadata"""
    try:
        return version('rephraser')
    except PackageNotFoundError:
        return 'unknown'


def sanitizeandmutateword(word: str) -> str:
    """
    Capitalize the first letter
    """
    if word[0] in undesirable_chars:
        word = word[1:]
    if word != '' and word[-1] in undesirable_chars:
      word = word[0:len(word)-1]
    if len(word) > 1:
        return word[0].capitalize() + word[1:]  # Preserve rest of case on capitalized acronyms
    else:
        return word.capitalize()

def collectall(state: list, depth: int, func_prefix: list) -> list:
    """
    Given a compiled DCT and state, return a list of all phrases (lists) of exactly a certain length/depth in titlecase
    """
    completedchains: list[list[str]] = []
    if DCT is None:
        raise RuntimeError("DCT is not initialized")

    state_key = ' '.join(state)
    if state_key not in DCT:
        return completedchains
    cstate_model = DCT[state_key].value
    if not func_prefix:
        func_prefix = list(state)
    if depth > 1:
        for nextword in cstate_model[0]:
            if nextword != END:
                nextstate = list(state[1:]) + [nextword]
                nextreach = collectall(nextstate, depth - 1, func_prefix + [nextword])
                if nextreach:
                    completedchains += nextreach
    else:
        # fix-up prefix words outside of loop because they won't need to be passed further
        mutated_prefix: list[str] = []
        for word in func_prefix:
            mutated_prefix.append(sanitizeandmutateword(word))
        for nextword in cstate_model[0]:
            if nextword not in [END, '']:
                mutated_word = sanitizeandmutateword(nextword)
                if mutated_word != '':
                    completedchains.append(mutated_prefix + [mutated_word])
    return completedchains

def workercollectall(func_mpqueue: mp.Queue, gpusaturated: bool = False) -> None:
    """Worker function to collect all chains of a certain depth, and output them in titlecase"""
    # Landing function for workers
    while True:
        try:
            arglist = func_mpqueue.get()
        except KeyboardInterrupt:
            break
        if len(arglist) == 3:
            state, depth, prefix = arglist
            if state == DONE:
                break
            outchains: list[list[str]] = collectall(state, depth, prefix)
            # output to STDOUT (outlist should be titlecase mutated, result should be titlecase with interspace)
            if not gpusaturated:
                for outlist in outchains:
                    # Titlecase with spaces
                    sys.stdout.write(' '.join(outlist) + '\n')
                    sys.stdout.flush()
            else:
                for outlist in outchains:
                    # Titlecase with spaces
                    sys.stdout.write(' '.join(outlist) + '\n')
                    # Titlecase without spaces
                    sys.stdout.write(''.join(outlist) + '\n')
                    # Lowercase with spaces
                    sys.stdout.write(' '.join(outlist).lower() + '\n')
                    # Lowercase without spaces
                    sys.stdout.write(''.join(outlist).lower() + '\n')
                    # First letter capitalized with spaces
                    sys.stdout.write(outlist[0] + ' ' + ' '.join(outlist[1:]).lower() + '\n')
                    # First letter capitalized without spaces
                    sys.stdout.write(outlist[0] + ''.join(outlist[1:]).lower() + '\n')
                    # Camelcase with spaces
                    sys.stdout.write(outlist[0].lower() + ' ' + ' '.join(outlist[1:]) + '\n')
                    # Camelcase without spaces
                    sys.stdout.write(outlist[0].lower() + ''.join(outlist[1:]) + '\n')
                    sys.stdout.flush()

def traverselikely(func_mpqueue: mp.Queue, state: tuple, depthremaining: int, batchdepth: int, func_prefix: list | None = None) -> None:
    """
    Traverse the Markov model in order of most likely next word,
    until a certain depth, at which point put work on the queue for workers to handle in bulk
    """
    # stateweights = [[weight, index], [weight, index]]
    stateweights: list[list[int]] = []
    # Sort and traverse from at least the most common start-points
    if func_prefix is None:
        func_prefix = []
    if DCT is None:
        return
    state_key = ' '.join(state)
    if state_key not in DCT:
        return
    cstate_model = DCT[state_key].value
    for weights in range(len(cstate_model[1])):
        if weights == 0:
            stateweights.append([cstate_model[1][weights], 0])
        else:
            stateweights.append([cstate_model[1][weights] - cstate_model[1][weights-1], weights])
    stateweights.sort(reverse=True)

    if depthremaining <= batchdepth:
        for weighted in stateweights:
            # cstate_model[0 = words][wordindex]
            nextword = cstate_model[0][weighted[1]]
            if nextword == END:
                continue
            nextstate = tuple(state[1:]) + (nextword,)
            # Parallelize below batchdepth
            func_mpqueue.put([nextstate, depthremaining - 1, func_prefix + [nextword]])
    else:
        for weighted in stateweights:
            # DCT[state][0 = words][wordindex]
            nextword = cstate_model[0][weighted[1]]
            if nextword == END:
                continue
            nextstate = tuple(state[1:]) + (nextword,)
            traverselikely(func_mpqueue, nextstate, depthremaining - 1, batchdepth, func_prefix + [nextword])

def main():
    global DCT_KEYS, DCT, mpqueue, worker_num  # noqa: PLW0602
    parser = argparse.ArgumentParser(prog='rephraser', description='Program for taking in either a model or corpus, and outputting markov chains of a specified word-length', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', '-v', action='version', version=f'%(prog)s {get_version()}')
    parser.add_argument('--model', '-m', required=True, help='Path to a saved model (make sure to set --ngrams if using 3grams) or where to save the model generated', default='')
    parser.add_argument('--ngrams', '-g', type=int, help='Number of words (n-grams) that make up a state in the Markov model, it is suggested to use 2 for large corpuses where the resulting model size might overrun RAM, and 3 for the better linguistic accuracy', choices=[2, 3], default=2)
    parser.add_argument('--corpus', '-c', help='Path to a corpus (file with sentences) to convert into a Markov model', default='')
    parser.add_argument('--corpusisdir', '-d', action='store_true', help='Handle the "--corpus" path as a directory, and create a model by walking all files inside', default=False)
    parser.add_argument('--words', '-w', type=int, help='Number of words in outputtable candidates', default=4)
    parser.add_argument('--workers', '-x', type=int, help='Manually specify the number of workers', default=mp.cpu_count() - 1)
    parser.add_argument('--freqlist', '-f', help='Path to a frequency list of *lowercase words*, one per line (E.g. Google 10k most common words), to use as start words. Warning: This is a n^2 operation, and may take a couple minutes to find all chain start-points (in order) depending on model * freqlist size.', default='')
    parser.add_argument('--gpusaturated', '-s', action='store_true', help='If hashcat is liable to be saturated with work, create "Basic8" permutations in the CPU workers, usually nets a little extra performance on fast hashes', default=False)
    parser.add_argument('--batchdepth', '-b', type=int, help='Number of ending words/recursions that should be handled in bulk by performing "collectall"', default=3)

    args = parser.parse_args()

    # Sanity checks, and model load/creation
    if args.corpus != '' and args.model != '':
        if args.corpusisdir and os.path.isdir(args.corpus):
            # Load multi-file corpus from --corpus
            COMBINED_MODEL = None
            for (dirpath, _, filenames) in os.walk(args.corpus):
                for filename in filenames:
                    sys.stderr.write('[REPHRASER] Loading ' + os.path.join(dirpath, filename) + ' ...')
                    with open(os.path.join(dirpath, filename), encoding='utf-8', errors='ignore') as f:
                        mmodel = markovify.Text(f, retain_original=False, state_size=args.ngrams)
                        if COMBINED_MODEL:
                            sys.stderr.write('[REPHRASER] Combining ' + os.path.join(dirpath, filename) + ' ...')
                            COMBINED_MODEL = markovify.combine(models=[COMBINED_MODEL, mmodel])
                        else:
                            COMBINED_MODEL = mmodel
            del mmodel
            if COMBINED_MODEL is None:
                sys.stderr.write('[REPHRASER] No files found in directory ' + args.corpus + ' Exiting!\n')
                sys.exit(1)
            COMBINED_MODEL.compile(inplace=True)
            keyvicompiler = keyvi.compiler.JsonDictionaryCompiler()
            DCT_KEYS.clear()
            for key in COMBINED_MODEL.chain.model:
                key_str = ' '.join(key)
                DCT_KEYS.add(key_str)
                keyvicompiler.add(key_str, json.dumps(COMBINED_MODEL.chain.model[key]))
            del COMBINED_MODEL
            keyvicompiler.compile()
            keyvicompiler.write_to_file(args.model)
            del keyvicompiler
            DCT = keyvi.dictionary.Dictionary(args.model)
        elif os.path.isfile(args.corpus):
            # Load single-file corpus from --corpus
            with open(args.corpus, encoding='utf-8') as f:
                mmodel = markovify.Text(f, retain_original=False, state_size=args.ngrams)
            mmodel.compile(inplace=True)
            keyvicompiler = keyvi.compiler.JsonDictionaryCompiler()
            DCT_KEYS.clear()
            for key in mmodel.chain.model:
                key_str = ' '.join(key)
                DCT_KEYS.add(key_str)
                keyvicompiler.Add(key_str, json.dumps(mmodel.chain.model[key]))
            del mmodel
            keyvicompiler.Compile()
            keyvicompiler.WriteToFile(args.model)
            del keyvicompiler
            DCT = keyvi.dictionary.Dictionary(args.model)
    elif args.model != '':
        # Load a saved model in a keyvi file
        if os.path.isfile(args.model):
            DCT_KEYS.clear()
            DCT = keyvi.dictionary.Dictionary(args.model)
            # Populate DCT_KEYS from the dictionary
            for key_str in DCT.keys():  # noqa: SIM118
                DCT_KEYS.add(key_str)
        else:
            sys.stderr.write('[REPHRASER] Couldn\'t find model at ' + args.model + '\n[REPHRASER] Exiting!\n')
            sys.exit(1)
    else:
        sys.stderr.write('[REPHRASER] You must specify either a corpus to scan and a model path to save in, or a pre-compiled model to load!\n[REPHRASER] Exiting!\n')
        sys.exit(1)

    if args.workers < 1:
        worker_num = 1
    else:
        worker_num = args.workers

    mpqueue = mp.Queue(MAXQUEUESIZE)
    GPUSATURATED = args.gpusaturated # Whether to create "Basic8" permutations in CPU workers
    # Spin up workers once and early
    worker_processes = []
    for i in range(worker_num):
        worker = mp.Process(target=workercollectall, args=(mpqueue, GPUSATURATED))
        worker.daemon = True
        worker.start()
        worker_processes.append(worker)

    # Change signal handling in only parent
    signal(SIGINT, sigint_handler)
    if args.freqlist != '':
        # Iterate on most-frequently used words input, as long as they in the model
        freqlist: list[str] = []
        if os.path.isfile(args.freqlist):
            with open(args.freqlist, encoding='utf-8', errors='ignore') as f:
                freqlist = f.read().split('\n')
        else:
            sys.stderr.write('[REPHRASER] Couldn\'t find freqlist at ' + args.freqlist + '\n[REPHRASER] Exiting!\n')
            sys.exit(1)

        freqtuplelists: list[list[tuple]] = []
        # Create array of arrays to hold keys corresponding to words in freqlist
        for freq in freqlist:
            freqtuplelists.append([])
        # Iterate through all markov chain keys, keeping those that are in our freqlist, in the order of freqlist
        if DCT is None:
            raise RuntimeError("DCT is not initialized")
        for key in DCT_KEYS:
            if key == f'{BEGIN} {BEGIN}' or key == f'{BEGIN} {BEGIN} {BEGIN}':
                continue
            if END in key:
                continue
            tuplekey = tuple(key.split(' ', args.ngrams - 1))
            if args.ngrams > 2 and BEGIN in tuplekey[1]:
                try:
                    foundindex = freqlist.index(tuplekey[2].lower())
                    freqtuplelists[foundindex].append(tuplekey)
                except ValueError:
                    pass
            elif BEGIN in tuplekey[0]:
                try:
                    foundindex = freqlist.index(tuplekey[1].lower())
                    freqtuplelists[foundindex].append(tuplekey)
                except ValueError:
                    pass
            else:
                try:
                    foundindex = freqlist.index(tuplekey[0].lower())
                    freqtuplelists[foundindex].append(tuplekey)
                except ValueError:
                    pass
        # No need for freqlist this point onward
        del freqlist
        # Iterate on keys, handling the indicated keys only
        for freqtuplelist in freqtuplelists:
            if not freqtuplelist:
                continue
            for tuplekey in freqtuplelist:
                prefix_normal = list(tuplekey)
                prefixmod = args.ngrams
                if tuplekey[0] == BEGIN:
                    prefixmod = args.ngrams - 1
                    prefix_normal = list(tuplekey[1:])
                if tuplekey[1] == BEGIN and args.ngrams > 2:
                    prefixmod = args.ngrams - 2
                    prefix_normal = list(tuplekey[2:])
                # Handle the common-case where the tuplekey puts us below the batchdepth
                if args.words - prefixmod <= args.batchdepth:
                    mpqueue.put([tuplekey, args.words - prefixmod - 1, prefix_normal])
                else:
                    traverselikely(mpqueue, tuplekey, args.words - prefixmod, args.batchdepth, prefix_normal)
    else:
        # Iterate on all keys in chain model, handling most likely key (start of sentence) first
        if DCT is None:
            raise RuntimeError("DCT is not initialized")
        if args.ngrams == 2:
            traverselikely(mpqueue, (BEGIN, BEGIN), args.words, args.batchdepth, [])
        elif args.ngrams == 3:
            traverselikely(mpqueue, (BEGIN, BEGIN, BEGIN), args.words, args.batchdepth, [])
        for key in DCT_KEYS:
            if key == f'{BEGIN} {BEGIN}' or key == f'{BEGIN} {BEGIN} {BEGIN}':
                continue
            # Need to convert string keys back into tuples for programmatic use
            tuplekey = tuple(key.split(' ', args.ngrams - 1))
            if END in tuplekey:
                continue
            prefix_normal = list(tuplekey)
            prefixmod = args.ngrams
            if tuplekey[0] == BEGIN:
                prefixmod = args.ngrams - 1
                prefix_normal = list(tuplekey[1:])
            if tuplekey[1] == BEGIN and args.ngrams > 2:
                prefixmod = args.ngrams - 2
                prefix_normal = list(tuplekey[2:])
            # Handle the common-case where the tuplekey puts us below the batchdepth
            if args.words - prefixmod <= args.batchdepth:
                mpqueue.put([tuplekey, args.words - prefixmod - 1, prefix_normal])
            else:
                traverselikely(mpqueue, tuplekey, args.words - prefixmod, args.batchdepth, prefix_normal)

    # Wrap up, send end-of-work signals to workers (one each)
    for worker in worker_processes:
        mpqueue.put([DONE, DONE, DONE])

    sys.stderr.write('\n[REPHRASER] Scheduler work completed! Waiting patiently for workers to finish queued work...\n')
    # Wait for workers to empty queue and hit done signals before killing parent process.
    for worker in worker_processes:
        worker.join()


if __name__ == '__main__':
    main()
