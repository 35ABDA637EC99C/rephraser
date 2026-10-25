"""
Fork from https://github.com/travco/rephraser
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from importlib.metadata import version
from signal import SIGINT, signal

import markovify  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]

BEGIN = '___BEGIN__'
END = '___END__'
DONE = '___DONE__'
undesirable_chars = [',','.',';',':','?','\'','"','"','"','"']

DCT = None  # Global dictionary for Markov model data
DCT_KEYS: set[str] = set()  # Global set of all dictionary keys for iteration
mpqueue = None # Work queue
MAXQUEUESIZE = 100000  # Number of work items reasonable to have on queue
worker_num = 0  # Will be changed before creating workers

# Memory governor state (parent-side): keeps predicted memory usage below a
# fraction of physical RAM by pausing the work producer while workers drain
# the queue. Output still flows while paused, so downstream consumers
# (hashcat) keep receiving input.
MEM_TARGET = 0.70  # Fraction of physical RAM we aim to stay under
TOTAL_MEM = 0  # Total physical memory in bytes (0 = unknown, governor disabled)
WORKITEM_BYTES = 0  # Predicted peak memory cost of a single queued work item
MEM_BASELINE = 0  # RSS of the process tree before any work is queued (model + idle workers)
_queued_bytes = 0.0  # Producer-side estimate of bytes of work currently sitting on the queue
_last_qsize = 0  # Queue size seen at the previous governor check, to detect consumed items
_start_time = 0.0  # monotonic time production started, for the startup grace period
START_GRACE_SECONDS = 30  # No throttling during the first N seconds so output starts fast
_avg_branch = 1.0  # Sampled average branching factor of the model (set by estimate_workitem_bytes)
_bytes_per_chain = 0.0  # Sampled average bytes per generated chain
worker_pids: list[int] = []  # PIDs of worker processes, for RSS summation
_put_counter = 0  # Cheap-check counter so we don't stat /proc on every put
_cached_rss_growth = 0  # Last measured tree RSS growth above the baseline


def get_available_memory() -> int:
    """
    Return currently available physical memory (excluding swap) in bytes, or 1000000000 (1GB)
    if it cannot be determined. Falls back to total physical RAM via sysconf,
    which also excludes swap but ignores current usage, so it can overestimate
    what is actually free.
    """
    try:
        with open('/proc/meminfo', encoding='ascii') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (ValueError, OSError, AttributeError):
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')

    return 1000000000


def process_rss(pid: int) -> int:
    """Return resident set size of a process in bytes, or 0 if unknown"""
    try:
        with open('/proc/' + str(pid) + '/status', encoding='ascii') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return 0
    return 0


def tree_rss() -> int:
    """Sum RSS of the parent (which holds the model) and all workers"""
    rss: int = process_rss(os.getpid())
    for pid in worker_pids:
        rss += process_rss(pid)
    return rss


def _measure_rss_growth() -> int:
    """Tree RSS growth above the pre-production baseline, floored at 0"""
    growth = tree_rss() - MEM_BASELINE
    return max(growth, 0)


def estimate_workitem_bytes(batchdepth: int) -> int:
    """
    Estimate the peak memory a single queued work item will cost when a worker
    expands it with collectall. Sampled from the model: average branching factor
    gives the expected number of chains (branching ** depth), each chain
    costing roughly its words' characters plus per-string object overhead.
    Deliberately conservative (2x safety factor) since underestimating causes OOM.
    Also stores the sampled stats in _avg_branch / _bytes_per_chain for use by
    item_bytes(), which prices each queued item by its own depth.
    """
    global _avg_branch, _bytes_per_chain
    if DCT is None or not DCT:
        return 0
    sampled_states = 0
    branch_total = 0
    wordlen_total = 0
    word_count = 0
    for key in DCT:
        words = DCT[key][0]
        branch_total += len(words)
        sampled_states += 1
        for word in words:
            wordlen_total += len(word)
            word_count += 1
        if sampled_states >= 20000:
            break
    if sampled_states == 0 or word_count == 0:
        return 0
    avg_branch = min(branch_total / sampled_states, 100.0)  # cap pathological branching
    avg_wordlen = wordlen_total / word_count
    _avg_branch = max(avg_branch, 1.0)
    _bytes_per_chain = batchdepth * (avg_wordlen + 50) + 100  # str object overhead
    expected_chains = max(avg_branch ** max(batchdepth, 1), 1.0)
    return int(expected_chains * _bytes_per_chain * 2)


def item_bytes(depth: int) -> float:
    """Estimated peak memory cost of one queued work item of the given depth"""
    if WORKITEM_BYTES <= 0:
        return 0.0
    expected_chains = max(_avg_branch ** max(depth, 1), 1.0)
    return expected_chains * _bytes_per_chain * 2


def _reconcile_queue_bytes(qsize: int) -> None:
    """Account for items workers consumed since the last check"""
    global _queued_bytes, _last_qsize
    consumed = _last_qsize - qsize
    if consumed > 0:
        _queued_bytes -= consumed * WORKITEM_BYTES
        if _queued_bytes < 0:
            _queued_bytes = 0.0
    _last_qsize = qsize


def current_qsize() -> int:
    """Queue length, or 0 if qsize() is unsupported on this platform"""
    if mpqueue is None:
        return 0
    try:
        return mpqueue.qsize()
    except NotImplementedError:
        return 0


def throttle() -> None:
    """
    Called before queueing work. Guarantees that no work item is queued while
    predicted memory usage is at/above the target: the check runs before every
    single put, using a cheap cached estimate (last measured RSS growth plus
    the tracked queue bytes). The expensive /proc re-measure happens every 32
    puts to keep overhead negligible, and inside the wait loop so blocked
    producers react as workers drain the queue. During the startup grace
    period the /proc re-measure runs on every put instead (still cheap), so
    the cached growth value starts out accurate.
    """
    global _put_counter, _cached_rss_growth
    _put_counter += 1
    if TOTAL_MEM <= 0 or WORKITEM_BYTES <= 0 or MEM_TARGET <= 0:
        return
    limit = TOTAL_MEM * MEM_TARGET
    in_grace = time.monotonic() - _start_time < START_GRACE_SECONDS
    if _put_counter % 32 == 0 or in_grace:
        _reconcile_queue_bytes(current_qsize())
        _cached_rss_growth = _measure_rss_growth()
    while _cached_rss_growth + int(_queued_bytes) >= limit:
        _reconcile_queue_bytes(current_qsize())
        _cached_rss_growth = _measure_rss_growth()
        time.sleep(0.002)


try:
    mp.set_start_method("fork")
except RuntimeError:
    sys.stderr.write('[REPHRASER] Fork start method not available, falling back to default.\n')

def sigint_handler(signal_received, frame):
  # Parent *should* be able to exit
  sys.stderr.write('[REPHRASER] SIGINT or CTRL-C detected. Attempting to exit gracefully...\n')
  last_work_item = mpqueue.get(block=False)  # ty: ignore[unresolved-attribute]
  sys.stderr.write('[REPHRASER] Next prefix in queue was: ' + repr(last_work_item)[2] + '\n')
  sys.exit(0)

def get_version() -> str:
    """Read the installed package version from distribution metadata"""
    
    return version('rephraser')


def sanitizeandmutateword(word: str) -> str:
    """
    Capitalize the first letter
    """
    if word == '':
        return ''
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
    cstate_model = DCT[state_key]
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
                    space_joined = ' '.join(outlist)
                    no_space = ''.join(outlist)
                    lower_space = space_joined.lower()
                    lower_no_space = no_space.lower()
                    first, *rest = outlist
                    rest_space = ' '.join(rest)
                    rest_no_space = ''.join(rest)
                    rest_lower_space = rest_space.lower()
                    rest_lower_no_space = rest_no_space.lower()
                    # Titlecase with spaces
                    sys.stdout.write(space_joined + '\n')
                    # Titlecase without spaces
                    sys.stdout.write(no_space + '\n')
                    # Lowercase with spaces
                    sys.stdout.write(lower_space + '\n')
                    # Lowercase without spaces
                    sys.stdout.write(lower_no_space + '\n')
                    # First letter capitalized with spaces
                    sys.stdout.write(first + ' ' + rest_lower_space + '\n')
                    # First letter capitalized without spaces
                    sys.stdout.write(first + rest_lower_no_space + '\n')
                    # Camelcase with spaces
                    sys.stdout.write(first.lower() + ' ' + rest_space + '\n')
                    # Camelcase without spaces
                    sys.stdout.write(first.lower() + rest_no_space + '\n')
                    sys.stdout.flush()

def traverselikely(func_mpqueue: mp.Queue, state: tuple, depthremaining: int, batchdepth: int, func_prefix: list | None = None) -> None:
    """
    Traverse the Markov model in order of most likely next word,
    until a certain depth, at which point put work on the queue for workers to handle in bulk
    """
    global _queued_bytes
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
    cstate_model = DCT[state_key]
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
            throttle()
            _queued_bytes += item_bytes(depthremaining - 1)
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
    global DCT_KEYS, DCT, mpqueue, worker_num, MEM_TARGET, TOTAL_MEM, WORKITEM_BYTES, worker_pids, MEM_BASELINE, _queued_bytes, _start_time  # noqa: PLW0602
    parser = argparse.ArgumentParser(prog='rephraser', description='Program for taking in either a model or corpus, and outputting markov chains of a specified word-length', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', '-v', action='version', version=f'%(prog)s {get_version()}')
    parser.add_argument('--model', '-m', required=True, help='Path to a saved model (gzip-compressed JSON, make sure to set --ngrams if using 3grams) or where to save the model generated', default='')
    parser.add_argument('--ngrams', '-g', type=int, help='Number of words (n-grams) that make up a state in the Markov model, it is suggested to use 2 for large corpuses where the resulting model size might overrun RAM, and 3 for the better linguistic accuracy', choices=[2, 3], default=2)
    parser.add_argument('--corpus', '-c', help='Path to a corpus (file with sentences) to convert into a Markov model', default='')
    parser.add_argument('--words', '-w', type=int, help='Number of words in outputtable candidates', default=1)
    parser.add_argument('--workers', '-x', type=int, help='Manually specify the number of workers', default=mp.cpu_count() - 1)
    parser.add_argument('--gpusaturated', '-s', action='store_true', help='If hashcat is liable to be saturated with work, create "Basic8" permutations in the CPU workers, usually nets a little extra performance on fast hashes', default=False)
    parser.add_argument('--batchdepth', '-b', type=int, help='Number of ending words/recursions that should be handled in bulk by performing "collectall"', default=3)
    parser.add_argument('--mem_limit', type=int, help='Percentage of physical RAM the memory governor aims to stay under while queueing work (0 disables the governor)', default=70)

    args = parser.parse_args()

    # Sanity checks, and model load/creation
    if args.corpus != '' and args.model != '':
        if os.path.isfile(args.corpus):
            # Load single-file corpus from --corpus
            with open(args.corpus, encoding='utf-8', errors='ignore') as f:
                mmodel = markovify.Text(f, retain_original=False, state_size=args.ngrams)
            mmodel.compile(inplace=True)
            DCT = {}
            DCT_KEYS.clear()
            for key in mmodel.chain.model:
                key_str = ' '.join(key)
                DCT_KEYS.add(key_str)
                DCT[key_str] = list(mmodel.chain.model[key])
            del mmodel
            with gzip.open(args.model, 'wt', encoding='utf-8') as f:
                json.dump(DCT, f)
        else:
            sys.stderr.write('[REPHRASER] Corpus path is neither a file nor a directory: ' + args.corpus + '\n[REPHRASER] Exiting!\n')
            sys.exit(1)
    elif args.model != '':
        # Load a saved model from gzip-compressed JSON file
        if os.path.isfile(args.model):
            DCT_KEYS.clear()
            with gzip.open(args.model, 'rt', encoding='utf-8') as f:
                DCT = json.load(f)
            # Populate DCT_KEYS from the dictionary
            for key_str in DCT:
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

    # Initialize the memory governor: predict per-work-item memory cost from the
    # model and track RSS of parent (model) plus workers against the RAM target.
    TOTAL_MEM = get_available_memory()
    if args.mem_limit > 0:
        MEM_TARGET = args.mem_limit / 100
    else:
        MEM_TARGET = 0  # disables throttle entirely
    WORKITEM_BYTES = estimate_workitem_bytes(args.batchdepth)
    worker_pids = [w.pid for w in worker_processes if w.pid is not None]
    # Baseline = RSS of parent (model) plus freshly started (idle) workers, so
    # the governor only reacts to memory growth caused by queued work.
    MEM_BASELINE = tree_rss()
    _start_time = time.monotonic()
    if TOTAL_MEM > 0:
        sys.stderr.write('[REPHRASER] Memory governor active: target ' + str(args.mem_limit) + '% of ' + str(TOTAL_MEM // (1024 * 1024)) + ' MiB available, estimated ' + str(WORKITEM_BYTES // 1024) + ' KiB per work item\n')
    else:
        sys.stderr.write('[REPHRASER] Could not determine available system memory, memory governor disabled\n')

    
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
            _queued_bytes += item_bytes(args.words - prefixmod - 1)
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
