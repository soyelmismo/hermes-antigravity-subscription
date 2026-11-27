"""Child-process entry points for the cross-process registry tests.

Kept in their own module so a ``spawn`` child imports only the registry, not
the test module and the client behind it.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Owner:
    pass


def hold_claim(store_dir, conversation_id, claimed, release, result):
    """Claim, report, then hold until told to release (or until killed)."""
    from session_registry import SessionRegistry

    reg = SessionRegistry(store_dir=store_dir)
    owner = Owner()
    result.put((os.getpid(), reg.claim(conversation_id, owner)))
    claimed.set()
    release.wait(30)
    reg.release_owner(owner)


def race_claim(store_dir, conversation_id, start, done, result):
    """Try once to claim at the same moment as the others; hold until done."""
    from session_registry import SessionRegistry

    reg = SessionRegistry(store_dir=store_dir)
    owner = Owner()
    start.wait(30)
    result.put(reg.claim(conversation_id, owner))
    done.wait(30)
    reg.release_owner(owner)


def lookup_history_len(store_dir, key, messages, result):
    """A fresh process looks the key up; reports what it found."""
    from session_registry import SessionRegistry

    reg = SessionRegistry(store_dir=store_dir)
    snap = reg.lookup(key, messages)
    result.put(None if snap is None else (snap.conversation_id, len(snap.history)))
