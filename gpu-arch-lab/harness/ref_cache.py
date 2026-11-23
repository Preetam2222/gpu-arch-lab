"""Independent pure-Python reference model (LRU / FIFO) used to cross-check the C++ simulator.

Deliberately written differently from the C++ (OrderedDict per set, no timestamps).
"""
from collections import OrderedDict


class RefCache:
    def __init__(self, size, assoc, line, policy="lru", classify=False):
        assert policy in ("lru", "fifo")
        self.assoc, self.line, self.policy = assoc, line, policy
        self.nsets = size // (line * assoc)
        self.sets = [OrderedDict() for _ in range(self.nsets)]  # line_addr -> dirty
        self.s = dict(reads=0, writes=0, read_hits=0, write_hits=0,
                      evictions=0, writebacks=0, compulsory=0, capacity=0, conflict=0)
        self.classify = classify
        self.seen = set()
        self.shadow = OrderedDict()
        self.nlines = self.nsets * assoc

    def _shadow(self, la):
        hit = la in self.shadow
        if hit:
            self.shadow.move_to_end(la)
        else:
            self.shadow[la] = True
            if len(self.shadow) > self.nlines:
                self.shadow.popitem(last=False)
        return hit

    def access(self, addr, is_write):
        """Returns (hit, dirty_victim_byte_addr_or_None)."""
        la = addr // self.line
        st = self.sets[la % self.nsets]
        self.s["writes" if is_write else "reads"] += 1
        first = fa_hit = None
        if self.classify:
            first = la not in self.seen
            self.seen.add(la)
            fa_hit = self._shadow(la)
        if la in st:
            if self.policy == "lru":
                st.move_to_end(la)
            if is_write:
                st[la] = True
            self.s["write_hits" if is_write else "read_hits"] += 1
            return True, None
        if self.classify:
            self.s["compulsory" if first else ("capacity" if not fa_hit else "conflict")] += 1
        victim = None
        if len(st) >= self.assoc:
            vla, vdirty = st.popitem(last=False)  # LRU end (lru) or oldest insert (fifo)
            self.s["evictions"] += 1
            if vdirty:
                self.s["writebacks"] += 1
                victim = vla * self.line
        st[la] = bool(is_write)
        return False, victim

    @property
    def hits(self):
        return self.s["read_hits"] + self.s["write_hits"]

    @property
    def accesses(self):
        return self.s["reads"] + self.s["writes"]


class RefHierarchy:
    def __init__(self, levels, policy="lru"):
        """levels: list of (size, assoc, line); level 0 is classified."""
        self.levels = [RefCache(*cfg, policy=policy, classify=(i == 0))
                       for i, cfg in enumerate(levels)]

    def access(self, addr, is_write, i=0):
        hit, victim = self.levels[i].access(addr, is_write)
        if i + 1 < len(self.levels):
            if victim is not None:
                self.access(victim, True, i + 1)
            if not hit:
                self.access(addr, False, i + 1)
