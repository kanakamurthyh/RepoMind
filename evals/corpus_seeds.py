"""Reviewable question seeds, selected independently of retrieval results.

These produce draft labels. A maintainer must approve gold spans and answers
before the external corpus can be used as a release quality gate.
"""

from __future__ import annotations

REPOSITORIES = [
    {"name": "pallets/itsdangerous", "split": "dev", "language": "python",
     "primary": "src/itsdangerous/signer.py", "secondary": "src/itsdangerous/serializer.py",
     "tests": "tests/test_itsdangerous/test_signer.py", "manifest": "pyproject.toml", "readme": "README.md", "license": "LICENSE.txt",
     "features": [
         ("How is a payload combined with its signature?", "    def sign(", "Signing appends the configured separator and the computed signature to the payload."),
         ("How are multiple secret keys tried during signature verification?", "    def verify_signature(self, value:", "Verification decodes the signature and tries secret keys in reverse order."),
         ("What happens if the signed input contains no separator?", "    def unsign(", "Unsigning rejects input without the separator with BadSignature."),
         ("How does the boolean validity helper handle invalid signatures?", "    def validate(", "The helper attempts unsigning and converts an invalid signature into a false result."),
     ]},
    {"name": "dbader/schedule", "split": "test", "language": "python",
     "primary": "schedule/__init__.py", "secondary": "test_schedule.py", "tests": "test_schedule.py",
     "manifest": "setup.py", "readme": "README.rst", "license": "LICENSE.txt",
     "features": [
         ("Does the pending-job runner replay every missed interval?", "    def run_pending(", "It deliberately does not replay all missed runs; it selects currently runnable jobs and executes them in sorted order."),
         ("How can a completed job request its own removal?", "    def _run_job(", "The scheduler removes the job when the return value is CancelJob or a CancelJob instance."),
         ("How are callback arguments bound when registering a job?", "    def do(", "Registration binds the callback and arguments with functools.partial before scheduling the next run."),
         ("What happens when the run-all helper receives a delay?", "    def run_all(", "It iterates a copy of the jobs, runs each, and sleeps for the requested delay between iterations."),
     ]},
    {"name": "encode/httpx", "split": "dev", "language": "python",
     "primary": "httpx/_client.py", "secondary": "httpx/_auth.py", "tests": "tests/client/test_auth.py",
     "manifest": "pyproject.toml", "readme": "README.md", "license": "LICENSE.md",
     "features": [
         ("How does synchronous request handling advance an authentication exchange?", "    def _send_handling_auth(", "It advances the auth-flow generator, sends requests, and feeds responses back until the flow finishes."),
         ("Where are response hooks invoked while following redirects?", "    def _send_handling_redirects(", "The redirect handler invokes response hooks after each individual request and before constructing the next redirect request."),
         ("How does the synchronous client reject use after it has been closed?", "    def send(", "The send method checks client state and rejects a closed client."),
         ("How does a single synchronous request choose and invoke its transport?", "    def _send_single_request(", "It selects the transport for the request URL, checks the request stream, and delegates the request to the transport."),
     ]},
    {"name": "mahmoud/boltons", "split": "test", "language": "python",
     "primary": "boltons/iterutils.py", "secondary": "boltons/cacheutils.py", "tests": "tests/test_iterutils.py",
     "manifest": "pyproject.toml", "readme": "README.md", "license": "LICENSE",
     "features": [
         ("How does chunking limit the number of groups returned?", "def chunked(", "The helper uses itertools.islice when a count is supplied; otherwise it consumes the chunk iterator."),
         ("How are incomplete final groups handled by the chunk iterator?", "def chunked_iter(", "The final group can be shorter; an optional fill value enables padding."),
         ("How does the list-returning flatten helper relate to its iterator variant?", "def flatten(", "It materializes the flatten iterator into a list."),
         ("How can recursive remapping customize traversal and entry transformation?", "def remap(", "It exposes traversal and transformation callbacks; exact callback and error behavior requires checking the cited implementation."),
     ]},
    {"name": "sindresorhus/p-limit", "split": "dev", "language": "javascript",
     "primary": "index.js", "secondary": "index.d.ts", "tests": "test.js", "manifest": "package.json", "readme": "readme.md", "license": "license",
     "features": [
         ("What prevents starting another queued task when the concurrency cap is reached?", "const resumeNext", "A queued task starts only when activeCount is below concurrency and the queue is nonempty."),
         ("After a task rejects, can a later queued task still start?", "const run =", "The task result is awaited inside a catch, after which next decrements the active count and resumes queued work."),
         ("How does clearing the queue behave when rejection on clearing is enabled?", "clearQueue:", "It rejects queued promises with an abort reason rather than merely clearing the queue."),
         ("How does changing the concurrency setting trigger more queued work?", "set(newConcurrency)", "It validates the new limit and schedules a microtask that resumes work while slots remain."),
     ]},
    {"name": "ai/nanoid", "split": "test", "language": "javascript",
     "primary": "index.js", "secondary": "non-secure/index.js", "tests": "test/index.test.js", "manifest": "package.json", "readme": "README.md", "license": "LICENSE",
     "features": [
         ("How are random buffers larger than the platform request limit filled?", "function fillRandom", "The buffer is filled in slices no larger than GET_RANDOM_LIMIT using crypto.getRandomValues."),
         ("How does direct random-byte generation reject a negative size?", "export function random", "It coerces the size to an integer and raises RangeError when negative."),
         ("How does a custom alphabet avoid biased character selection?", "export function customRandom", "It rejects random bytes above a cutoff divisible by alphabet length; power-of-two alphabets use a mask fast path."),
         ("When does custom-alphabet generation fall back from its byte-pool fast path?", "export function customAlphabet", "Unsupported alphabet types, empty or oversized alphabets, and non-single-byte characters use the general customRandom path."),
     ]},
    {"name": "lukeed/clsx", "split": "dev", "language": "javascript",
     "primary": "src/index.js", "secondary": "src/lite.js", "tests": "test/index.js", "manifest": "package.json", "readme": "readme.md", "license": "license",
     "features": [
         ("How are string and numeric arguments converted into class text?", "function toVal", "String and numeric values are appended to the output string."),
         ("How are nested arrays of class names flattened?", "if (Array.isArray", "Truthy nested values are recursively processed and separated with spaces when they yield text."),
         ("Which object keys become class names?", "for (y in mix)", "Keys whose values are truthy are appended to the class string."),
         ("How are false-like top-level arguments handled?", "export function clsx", "The argument loop processes only truthy arguments and joins nonempty conversions with spaces."),
     ]},
    {"name": "developit/mitt", "split": "test", "language": "typescript",
     "primary": "src/index.ts", "secondary": "test/test-types-compilation.ts", "tests": "test/index_test.ts",
     "manifest": "package.json", "readme": "README.md", "license": "LICENSE",
     "features": [
         ("Where does a new emitter keep registered handlers when no map is supplied?", "all = all || new Map", "It creates a new Map and exposes it through the returned emitter."),
         ("How does subscribing append to an existing event handler list?", "\t\ton<Key", "The subscription method pushes into an existing handler array or creates a new array for the event type."),
         ("What happens when an unsubscribe call omits the handler?", "\t\toff<Key", "It replaces the event type's handler list with an empty array."),
         ("In what order do typed-event handlers and wildcard handlers run?", "\t\temit<Key", "The emitter invokes copied type-specific handlers before copied wildcard handlers; wildcard handlers receive the type and payload."),
     ]},
]
