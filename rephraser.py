"""
Fork from https://github.com/travco/rephraser
"""
from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import os
import sys
import time
from functools import lru_cache
from importlib.metadata import version
from signal import SIGINT, signal

import markovify  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]
import orjson

BEGIN = '___BEGIN__'
END = '___END__'
DONE = '___DONE__'
undesirable_chars: tuple[str, ...] = (',','.',';',':','?','\'','"','`')

DCT = None  # Global dictionary for Markov model data
mpqueue = None # Work queue
MAXQUEUESIZE = 100000  # Number of work items reasonable to have on queue
worker_num = 0  # Will be changed before creating workers

# Memory governor state (parent-side): keeps the combined private memory growth
# of all workers under a fraction of physical RAM by pausing the work producer
# while workers process and drain their items. Output still flows while paused,
# so downstream consumers (hashcat) keep receiving input.
MEM_LIMIT_BYTES = 0  # Hard byte limit for combined worker RSS growth (0 = governor disabled)
TOTAL_MEM = 0  # Total physical memory in bytes (0 = unknown, governor disabled)
WORKITEM_BYTES = 0  # Predicted peak memory cost of a single queued work item
_avg_branch = 1.0  # Sampled average branching factor of the model (set by estimate_workitem_bytes)
_bytes_per_chain = 0.0  # Sampled average bytes per generated chain
worker_pids: list[int] = []  # PIDs of worker processes
_worker_base_rss: dict[int, int] = {}  # Idle RSS per worker, so only private growth counts
_cached_extra_rss = 0  # Last measured combined worker RSS growth
_last_measure_time = 0.0  # monotonic time of the last measurement (0 = never)
OOM_GUARD_THRESHOLD = 0.95  # Hard cap: no memory-growing work at 95% system RAM use, regardless of --mem_limit
OOM_STALL_ABORT_SECONDS = 60  # Give up remaining work if pinned at the threshold this long
_oom_cache_time = 0.0  # monotonic time of the last MemAvailable sample (0 = never)
_oom_cache_pressured = False  # Last sampled OOM-guard verdict


class WorkAborted(Exception):
    """Raised when the OOM guard gives up on the remaining work to avoid an OOM."""


def get_total_memory() -> int:
    """
    Return total physical memory in bytes, or 0 if it cannot be determined.
    """
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError):
        return 0


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


def worker_extra_rss(max_age: float = 0.05) -> int:
    """
    Sum of private RSS growth across all workers: each worker's current RSS
    minus its idle baseline. Workers are forked from the parent, so their RSS
    includes shared model pages; subtracting each worker's own baseline keeps
    those shared pages from being counted once per worker. Exited workers
    report RSS 0 and contribute nothing.

    Measurements are cached for max_age seconds so per-put calls are cheap;
    pass max_age=0 to force a fresh read (used inside the throttle wait loop).
    """
    global _cached_extra_rss, _last_measure_time
    now = time.monotonic()
    if _last_measure_time > 0 and now - _last_measure_time < max_age:
        return _cached_extra_rss
    total = 0
    for pid, base in _worker_base_rss.items():
        rss = process_rss(pid)
        if rss > base:
            total += rss - base
    _cached_extra_rss = total
    _last_measure_time = now
    return total


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
        if sampled_states >= 2000:
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


def memory_pressured(max_age: float = 0.25) -> bool:
    """
    True when system-wide memory use is at/above the OOM guard threshold
    (95% of physical RAM), regardless of --mem_limit. Measured from
    /proc/meminfo (MemAvailable vs MemTotal), cached for max_age seconds so
    per-put calls are cheap; pass max_age=0 to force a fresh read (used
    inside the stall wait loop).
    """
    global _oom_cache_time, _oom_cache_pressured
    now = time.monotonic()
    if _oom_cache_time > 0 and now - _oom_cache_time < max_age:
        return _oom_cache_pressured
    pressured = False
    try:
        with open('/proc/meminfo', encoding='ascii') as f:
            total = 0
            available = 0
            for line in f:
                if line.startswith('MemTotal:'):
                    total = int(line.split()[1]) * 1024
                elif line.startswith('MemAvailable:'):
                    available = int(line.split()[1]) * 1024
                if total and available:
                    break
            pressured = total > 0 and available <= total * (1 - OOM_GUARD_THRESHOLD)
    except (OSError, ValueError):
        pressured = False
    _oom_cache_pressured = pressured
    _oom_cache_time = now
    return pressured


def throttle(predicted_bytes: float) -> None:
    """
    Called before queueing work. Two guards, in order:

    1. OOM guard (always on): while system-wide memory use is at/above 95%
       of physical RAM, no work is queued. If the system stays pinned there
       for OOM_STALL_ABORT_SECONDS, remaining work is abandoned (WorkAborted)
       rather than risking an out-of-memory kill.
    2. Memory governor (if enabled via --mem_limit): blocks while the measured
       private RSS growth of all workers plus the predicted peak cost of the
       new work item fits under the memory limit.

    Checks use cached measurements on the hot path; re-measure fresh only
    while waiting, so draining workers unblock a blocked producer.
    """
    if memory_pressured():
        stalled_since = time.monotonic()
        while memory_pressured(max_age=0):
            if time.monotonic() - stalled_since > OOM_STALL_ABORT_SECONDS:
                sys.stderr.write('[REPHRASER] Memory pinned at ' + str(int(OOM_GUARD_THRESHOLD * 100)) + '%+ for ' + str(OOM_STALL_ABORT_SECONDS) + 's; abandoning remaining work to avoid out-of-memory\n')
                raise WorkAborted()
            time.sleep(0.25)
    if MEM_LIMIT_BYTES <= 0 or WORKITEM_BYTES <= 0:
        return
    # Cheap cached check on the hot path; re-measure fresh only while
    # waiting, so draining workers unblock a blocked producer.
    if worker_extra_rss() + predicted_bytes < MEM_LIMIT_BYTES:
        return
    while worker_extra_rss(max_age=0) + predicted_bytes >= MEM_LIMIT_BYTES:
        time.sleep(0.05)


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


@lru_cache(maxsize=65536)  # Model vocabulary is small relative to chain count; bounded to cap memory
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

def collectall(state: list, depth: int, func_prefix: list):
    """
    Given a compiled DCT and state, yield phrases (lists) of exactly a certain
    length/depth in titlecase. A generator so a work item with an enormous
    number of chains never materializes them all in memory at once.
    """
    if DCT is None:
        raise RuntimeError("DCT is not initialized")

    state_key = ' '.join(state)
    cstate_model = DCT.get(state_key)
    if cstate_model is None:
        return
    if not func_prefix:
        func_prefix = list(state)
    if depth > 1:
        for nextword in cstate_model[0]:
            if nextword != END:
                nextstate = list(state[1:]) + [nextword]
                yield from collectall(nextstate, depth - 1, func_prefix + [nextword])
    else:
        # fix-up prefix words outside of loop because they won't need to be passed further
        mutated_prefix: list[str] = []
        for word in func_prefix:
            mutated_prefix.append(sanitizeandmutateword(word))
        for nextword in cstate_model[0]:
            if nextword not in (END, ''):
                mutated_word = sanitizeandmutateword(nextword)
                if mutated_word != '':
                    yield mutated_prefix + [mutated_word]

def workercollectall(func_mpqueue: mp.Queue, gpusaturated: bool = False, flush_lines: int = 1) -> None:
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
            # Stream chains out of the collectall generator through a bounded
            # line buffer: memory stays O(buffer), not O(chains in item), and
            # gpusaturated items never hold 8x output in memory at once.
            lines: list[str] = []

            def flushlines(buf: list[str] = lines) -> None:
                if buf:
                    sys.stdout.write(''.join(buf))
                    sys.stdout.flush()
                    buf.clear()

            for outlist in collectall(state, depth, prefix):
                if not gpusaturated:
                    lines.append(' '.join(outlist) + '\n')
                else:
                    first, *rest = outlist
                    rest_space = ' '.join(rest)
                    first_lower = first.lower()
                    rest_lower_space = rest_space.lower()
                    # Model tokens never contain spaces, so the no-space
                    # lowercase variants derive from the spaced ones for free.
                    rest_lower_no_space = rest_lower_space.replace(' ', '')
                    space_joined = first + ' ' + rest_space
                    rest_no_space = ''.join(rest)
                    no_space = first + rest_no_space
                    space_joined_lower = space_joined.lower()
                    no_space_lower = space_joined_lower.replace(' ', '')
                    # Titlecase with spaces
                    lines.append(space_joined + '\n')
                    # Titlecase without spaces
                    lines.append(no_space + '\n')
                    # Lowercase with spaces
                    lines.append(space_joined_lower + '\n')
                    # Lowercase without spaces
                    lines.append(no_space_lower + '\n')
                    # First letter capitalized with spaces
                    lines.append(first + ' ' + rest_lower_space + '\n')
                    # First letter capitalized without spaces
                    lines.append(first + rest_lower_no_space + '\n')
                    # Camelcase with spaces
                    lines.append(first_lower + ' ' + rest_space + '\n')
                    # Camelcase without spaces
                    lines.append(first_lower + rest_no_space + '\n')
                if len(lines) >= flush_lines:
                    flushlines()
            flushlines()

def traverselikely(func_mpqueue: mp.Queue, state: tuple, depthremaining: int, batchdepth: int, func_prefix: list | None = None) -> None:
    """
    Traverse the Markov model in order of most likely next word,
    until a certain depth, at which point put work on the queue for workers to handle in bulk
    """
    # Sort and traverse from at least the most common start-points
    if func_prefix is None:
        func_prefix = []
    if DCT is None:
        return
    state_key = ' '.join(state)
    if state_key not in DCT:
        return
    cstate_model = DCT[state_key]
    # Sort word indices by their standalone weight (cumulative minus previous)
    weights = cstate_model[1]
    stateweights = sorted(range(len(weights)), key=lambda i: weights[i] - (weights[i - 1] if i else 0), reverse=True)

    if depthremaining <= batchdepth:
        for wordindex in stateweights:
            # cstate_model[0 = words][wordindex]
            nextword = cstate_model[0][wordindex]
            if nextword == END:
                continue
            nextstate = tuple(state[1:]) + (nextword,)
            # Parallelize below batchdepth; wait until worker memory headroom
            # allows this item's predicted peak cost.
            throttle(item_bytes(depthremaining - 1))
            func_mpqueue.put([nextstate, depthremaining - 1, func_prefix + [nextword]])
    else:
        for wordindex in stateweights:
            # DCT[state][0 = words][wordindex]
            nextword = cstate_model[0][wordindex]
            if nextword == END:
                continue
            nextstate = tuple(state[1:]) + (nextword,)
            traverselikely(func_mpqueue, nextstate, depthremaining - 1, batchdepth, func_prefix + [nextword])

def main():
    global DCT, mpqueue, worker_num, MEM_LIMIT_BYTES, TOTAL_MEM, WORKITEM_BYTES, worker_pids, _worker_base_rss
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
    parser.add_argument('--flush_lines', type=int, help='Number of output lines workers buffer before writing to stdout in a single batch', default=5000)

    args = parser.parse_args()

    # Sanity checks, and model load/creation
    if args.corpus != '' and args.model != '':
        if os.path.isfile(args.corpus):
            # Load single-file corpus from --corpus
            with open(args.corpus, encoding='utf-8', errors='ignore') as f:
                mmodel = markovify.Text(f, retain_original=False, state_size=args.ngrams)
            mmodel.compile(inplace=True)
            DCT = {}
            for key in mmodel.chain.model:
                DCT[' '.join(key)] = list(mmodel.chain.model[key])
            del mmodel
            with gzip.open(args.model, 'wb') as f:
                f.write(orjson.dumps(DCT))
        else:
            sys.stderr.write('[REPHRASER] Corpus path is neither a file nor a directory: ' + args.corpus + '\n[REPHRASER] Exiting!\n')
            sys.exit(1)
    elif args.model != '':
        # Load a saved model from gzip-compressed JSON file
        if os.path.isfile(args.model):
            with gzip.open(args.model, 'rb') as f:
                DCT = orjson.loads(f.read())
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

    # Initialize the memory governor sizing before creating the queue, so the
    # queue cap can be derived from the per-work-item memory estimate.
    TOTAL_MEM = get_total_memory()
    WORKITEM_BYTES = estimate_workitem_bytes(args.batchdepth)
    if args.mem_limit > 0 and TOTAL_MEM > 0 and WORKITEM_BYTES > 0:
        MEM_LIMIT_BYTES = int(TOTAL_MEM * args.mem_limit / 100)
    else:
        MEM_LIMIT_BYTES = 0  # disables the governor
    queue_capacity = MAXQUEUESIZE
    if MEM_LIMIT_BYTES > 0:
        # Hard safety net: cap the queue so buffered work items can never
        # exceed a quarter of the limit, leaving the rest for in-flight
        # expansion inside workers.
        queue_capacity = max(1, min(MAXQUEUESIZE, (MEM_LIMIT_BYTES // 4) // WORKITEM_BYTES))
        if queue_capacity < MAXQUEUESIZE:
            sys.stderr.write('[REPHRASER] Queue capacity limited to ' + str(queue_capacity) + ' items by memory governor\n')
    mpqueue = mp.Queue(queue_capacity)
    GPUSATURATED = args.gpusaturated # Whether to create "Basic8" permutations in CPU workers
    # Spin up workers once and early
    worker_processes = []
    for i in range(worker_num):
        worker = mp.Process(target=workercollectall, args=(mpqueue, GPUSATURATED, args.flush_lines))
        worker.daemon = True
        worker.start()
        worker_processes.append(worker)

    # Change signal handling in only parent
    signal(SIGINT, sigint_handler)

    # Initialize the memory governor: each worker's idle RSS becomes its
    # baseline, so only private growth from processing work counts toward
    # the limit (shared, copy-on-write model pages are not double-counted).
    worker_pids = [w.pid for w in worker_processes if w.pid is not None]
    _worker_base_rss = {pid: process_rss(pid) for pid in worker_pids}
    if MEM_LIMIT_BYTES > 0:
        sys.stderr.write('[REPHRASER] Memory governor active: workers limited to ' + str(args.mem_limit) + '% of ' + str(TOTAL_MEM // (1024 * 1024)) + ' MiB RAM, estimated ' + str(WORKITEM_BYTES // 1024) + ' KiB per work item\n')
    else:
        sys.stderr.write('[REPHRASER] Memory governor disabled\n')

    
    # Iterate on all keys in chain model, handling most likely key (start of sentence) first
    if DCT is None:
        raise RuntimeError("DCT is not initialized")
    try:
        if args.ngrams == 2:
            traverselikely(mpqueue, (BEGIN, BEGIN), args.words, args.batchdepth, [])
        elif args.ngrams == 3:
            traverselikely(mpqueue, (BEGIN, BEGIN, BEGIN), args.words, args.batchdepth, [])
        for key in DCT:
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
                throttle(item_bytes(args.words - prefixmod - 1))
                mpqueue.put([tuplekey, args.words - prefixmod - 1, prefix_normal])
            else:
                traverselikely(mpqueue, tuplekey, args.words - prefixmod, args.batchdepth, prefix_normal)
    except WorkAborted:
        pass  # Workers already drain their queued work; output continues below

    # Wrap up, send end-of-work signals to workers (one each)
    for worker in worker_processes:
        mpqueue.put([DONE, DONE, DONE])

    sys.stderr.write('\n[REPHRASER] Scheduler work completed! Waiting patiently for workers to finish queued work...\n')
    # Wait for workers to empty queue and hit done signals before killing parent process.
    for worker in worker_processes:
        worker.join()


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        sys.stderr.write('[REPHRASER] Broken pipe encountered. Exiting.\n')