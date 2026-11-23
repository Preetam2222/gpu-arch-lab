// cachesim: trace-driven multi-level cache simulator.
//   cachesim --l1 32K:8:64 [--l2 256K:8:64 ...] [--policy lru|fifo|random]
//            [--trace file] [--no-classify]
// Trace: one access per line, "R <addr>" or "W <addr>" (hex 0x.. or decimal);
// '#' starts a comment. Reads stdin when --trace is omitted. JSON on stdout.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache.hpp"

static uint64_t parse_size(const std::string& s) {
    char* end = nullptr;
    uint64_t v = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str()) throw std::invalid_argument("bad number: " + s);
    switch (*end) {
        case 'K': case 'k': v <<= 10; break;
        case 'M': case 'm': v <<= 20; break;
        case 'G': case 'g': v <<= 30; break;
        case '\0': break;
        default: throw std::invalid_argument("bad size suffix: " + s);
    }
    return v;
}

static CacheConfig parse_level(const std::string& spec, Policy p) {
    CacheConfig c;
    size_t a = spec.find(':'), b = spec.find(':', a == std::string::npos ? 0 : a + 1);
    if (a == std::string::npos || b == std::string::npos)
        throw std::invalid_argument("level spec must be SIZE:ASSOC:LINE, got " + spec);
    c.size = parse_size(spec.substr(0, a));
    c.assoc = parse_size(spec.substr(a + 1, b - a - 1));
    c.line = parse_size(spec.substr(b + 1));
    c.policy = p;
    return c;
}

struct Hierarchy {
    std::vector<std::unique_ptr<Cache>> levels;

    // A miss at level i fetches from level i+1; a dirty victim is written to i+1.
    void access(size_t i, uint64_t addr, bool is_write) {
        AccessResult r = levels[i]->access(addr, is_write);
        if (i + 1 >= levels.size()) return;
        if (r.evicted_dirty) access(i + 1, r.victim_addr, true);
        if (!r.hit) access(i + 1, addr, false);
    }
};

int main(int argc, char** argv) {
    try {
        std::vector<std::string> specs;
        Policy policy = Policy::LRU;
        std::string trace_path;
        bool classify = true;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };
            if (a == "--l1" || a == "--l2" || a == "--l3") {
                size_t idx = a[3] - '1';
                if (specs.size() <= idx) specs.resize(idx + 1);
                specs[idx] = next();
            } else if (a == "--policy") policy = parse_policy(next());
            else if (a == "--trace") trace_path = next();
            else if (a == "--no-classify") classify = false;
            else throw std::invalid_argument("unknown argument: " + a);
        }
        if (specs.empty() || specs[0].empty()) throw std::invalid_argument("--l1 is required");

        Hierarchy h;
        for (size_t i = 0; i < specs.size(); ++i) {
            if (specs[i].empty()) throw std::invalid_argument("levels must be contiguous (l1, l2, l3)");
            h.levels.emplace_back(new Cache(parse_level(specs[i], policy), classify && i == 0));
        }

        FILE* f = trace_path.empty() ? stdin : std::fopen(trace_path.c_str(), "r");
        if (!f) throw std::runtime_error("cannot open trace: " + trace_path);
        char buf[256];
        uint64_t lineno = 0;
        while (std::fgets(buf, sizeof buf, f)) {
            ++lineno;
            char* p = buf;
            while (*p == ' ' || *p == '\t') ++p;
            if (*p == '#' || *p == '\n' || *p == '\0') continue;
            char op = *p;
            if (op != 'R' && op != 'W' && op != 'r' && op != 'w')
                throw std::runtime_error("bad op at line " + std::to_string(lineno));
            char* end = nullptr;
            uint64_t addr = std::strtoull(p + 1, &end, 0);
            if (end == p + 1) throw std::runtime_error("bad address at line " + std::to_string(lineno));
            h.access(0, addr, op == 'W' || op == 'w');
        }
        if (f != stdin) std::fclose(f);

        std::printf("{\"policy\":\"%s\",\"levels\":[", policy_name(policy));
        for (size_t i = 0; i < h.levels.size(); ++i) {
            const Cache& c = *h.levels[i];
            const CacheStats& s = c.stats();
            double mr = s.accesses() ? double(s.misses()) / s.accesses() : 0.0;
            std::printf(
                "%s{\"level\":%zu,\"size\":%llu,\"assoc\":%llu,\"line\":%llu,\"sets\":%llu,"
                "\"accesses\":%llu,\"reads\":%llu,\"writes\":%llu,\"read_hits\":%llu,"
                "\"write_hits\":%llu,\"hits\":%llu,\"misses\":%llu,"
                "\"miss_rate\":%.6f,\"evictions\":%llu,\"writebacks\":%llu",
                i ? "," : "", i + 1, (unsigned long long)c.config().size,
                (unsigned long long)c.config().assoc, (unsigned long long)c.config().line,
                (unsigned long long)c.num_sets(), (unsigned long long)s.accesses(),
                (unsigned long long)s.reads, (unsigned long long)s.writes,
                (unsigned long long)s.read_hits, (unsigned long long)s.write_hits,
                (unsigned long long)s.hits(), (unsigned long long)s.misses(), mr,
                (unsigned long long)s.evictions, (unsigned long long)s.writebacks);
            if (classify && i == 0)
                std::printf(",\"compulsory\":%llu,\"capacity\":%llu,\"conflict\":%llu",
                            (unsigned long long)s.compulsory, (unsigned long long)s.capacity,
                            (unsigned long long)s.conflict);
            std::printf("}");
        }
        std::printf("]}\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
