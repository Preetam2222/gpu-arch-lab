// Set-associative, write-back / write-allocate cache model with optional
// 3C (compulsory / capacity / conflict) miss classification.
#pragma once
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum class Policy { LRU, FIFO, RANDOM };

Policy parse_policy(const std::string& s);
const char* policy_name(Policy p);

struct CacheConfig {
    uint64_t size = 32 * 1024;  // bytes
    uint64_t assoc = 8;         // ways
    uint64_t line = 64;         // bytes
    Policy policy = Policy::LRU;
};

struct CacheStats {
    uint64_t reads = 0, writes = 0;
    uint64_t read_hits = 0, write_hits = 0;
    uint64_t evictions = 0, writebacks = 0;
    uint64_t compulsory = 0, capacity = 0, conflict = 0;  // only if classify
    uint64_t accesses() const { return reads + writes; }
    uint64_t hits() const { return read_hits + write_hits; }
    uint64_t misses() const { return accesses() - hits(); }
};

struct AccessResult {
    bool hit = false;
    bool evicted_dirty = false;
    uint64_t victim_addr = 0;  // byte address of evicted dirty line
};

class Cache {
public:
    // Throws std::invalid_argument on non-power-of-two / inconsistent geometry.
    explicit Cache(const CacheConfig& cfg, bool classify = false);

    AccessResult access(uint64_t addr, bool is_write);
    const CacheStats& stats() const { return stats_; }
    const CacheConfig& config() const { return cfg_; }
    uint64_t num_sets() const { return sets_; }

private:
    struct Line {
        bool valid = false, dirty = false;
        uint64_t tag = 0, last_use = 0, inserted = 0;
    };

    bool shadow_access(uint64_t line_addr);  // fully-assoc LRU of equal capacity
    uint64_t next_rand();

    CacheConfig cfg_;
    uint64_t sets_, off_bits_, set_bits_;
    std::vector<Line> lines_;
    CacheStats stats_;
    uint64_t clock_ = 0, rng_ = 0x9E3779B97F4A7C15ull;

    bool classify_;
    std::unordered_set<uint64_t> seen_;
    std::list<uint64_t> shadow_lru_;  // front = MRU
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> shadow_map_;
};
