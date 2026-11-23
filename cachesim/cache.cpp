#include "cache.hpp"

#include <stdexcept>

static bool is_pow2(uint64_t x) { return x && !(x & (x - 1)); }
static uint64_t log2u(uint64_t x) {
    uint64_t n = 0;
    while (x > 1) { x >>= 1; ++n; }
    return n;
}

Policy parse_policy(const std::string& s) {
    if (s == "lru") return Policy::LRU;
    if (s == "fifo") return Policy::FIFO;
    if (s == "random") return Policy::RANDOM;
    throw std::invalid_argument("unknown policy: " + s);
}

const char* policy_name(Policy p) {
    switch (p) {
        case Policy::LRU: return "lru";
        case Policy::FIFO: return "fifo";
        default: return "random";
    }
}

Cache::Cache(const CacheConfig& cfg, bool classify) : cfg_(cfg), classify_(classify) {
    if (!is_pow2(cfg.line) || !is_pow2(cfg.assoc) || !is_pow2(cfg.size))
        throw std::invalid_argument("size, assoc and line must be powers of two");
    if (cfg.size < cfg.line * cfg.assoc)
        throw std::invalid_argument("size must be >= line * assoc");
    sets_ = cfg.size / (cfg.line * cfg.assoc);
    off_bits_ = log2u(cfg.line);
    set_bits_ = log2u(sets_);
    lines_.resize(sets_ * cfg.assoc);
}

uint64_t Cache::next_rand() {  // xorshift64, fixed seed => reproducible
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 7;
    rng_ ^= rng_ << 17;
    return rng_;
}

bool Cache::shadow_access(uint64_t la) {
    auto it = shadow_map_.find(la);
    if (it != shadow_map_.end()) {
        shadow_lru_.splice(shadow_lru_.begin(), shadow_lru_, it->second);
        return true;
    }
    shadow_lru_.push_front(la);
    shadow_map_[la] = shadow_lru_.begin();
    if (shadow_lru_.size() > sets_ * cfg_.assoc) {
        shadow_map_.erase(shadow_lru_.back());
        shadow_lru_.pop_back();
    }
    return false;
}

AccessResult Cache::access(uint64_t addr, bool is_write) {
    ++clock_;
    const uint64_t la = addr >> off_bits_;
    const uint64_t set = la & (sets_ - 1);
    const uint64_t tag = la >> set_bits_;
    Line* base = &lines_[set * cfg_.assoc];

    (is_write ? stats_.writes : stats_.reads)++;

    bool first_touch = false, fa_hit = false;
    if (classify_) {
        first_touch = seen_.insert(la).second;
        fa_hit = shadow_access(la);
    }

    AccessResult res;
    for (uint64_t w = 0; w < cfg_.assoc; ++w) {
        Line& l = base[w];
        if (l.valid && l.tag == tag) {
            if (cfg_.policy == Policy::LRU) l.last_use = clock_;
            if (is_write) l.dirty = true;
            (is_write ? stats_.write_hits : stats_.read_hits)++;
            res.hit = true;
            return res;
        }
    }

    if (classify_) {
        if (first_touch) ++stats_.compulsory;
        else if (!fa_hit) ++stats_.capacity;
        else ++stats_.conflict;
    }

    // Victim: first invalid way, else by policy.
    Line* victim = nullptr;
    for (uint64_t w = 0; w < cfg_.assoc && !victim; ++w)
        if (!base[w].valid) victim = &base[w];
    if (!victim) {
        if (cfg_.policy == Policy::RANDOM) {
            victim = &base[next_rand() % cfg_.assoc];
        } else {
            victim = &base[0];
            for (uint64_t w = 1; w < cfg_.assoc; ++w) {
                uint64_t a = cfg_.policy == Policy::LRU ? base[w].last_use : base[w].inserted;
                uint64_t b = cfg_.policy == Policy::LRU ? victim->last_use : victim->inserted;
                if (a < b) victim = &base[w];
            }
        }
        ++stats_.evictions;
        if (victim->dirty) {
            ++stats_.writebacks;
            res.evicted_dirty = true;
            res.victim_addr = ((victim->tag << set_bits_) | set) << off_bits_;
        }
    }
    victim->valid = true;
    victim->dirty = is_write;  // write-allocate
    victim->tag = tag;
    victim->last_use = victim->inserted = clock_;
    return res;
}
