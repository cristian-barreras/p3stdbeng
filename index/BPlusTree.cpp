#include "index/BPlusTree.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace bptree {

namespace {

// release — unpins a frame without disturbing `error`, which on failure
// paths already carries the real cause. Parameters: manager — the owner;
// frame — the pin to release; was_dirty — whether the page changed.
// Returns: nothing; pin failures here are swallowed by design.
void release(bufman::BufferManager& manager, std::size_t frame,
             bool was_dirty) {
    std::string unpin_error;
    manager.unpin(frame, was_dirty, unpin_error);
}

constexpr std::uint32_t kBptMagic = 0x31545042u; // "BPT1"

// Header block field offsets.
constexpr std::size_t kMagicOffset = 0;
constexpr std::size_t kKeyTypeOffset = 4;
constexpr std::size_t kKeyWidthOffset = 8;
constexpr std::size_t kRootOffset = 12;
constexpr std::size_t kFirstLeafOffset = 16;
constexpr std::size_t kLevelsOffset = 20;
constexpr std::size_t kLastBlockOffset = 24;
constexpr std::size_t kFreedHeadOffset = 28;
constexpr std::size_t kEntryCountOffset = 32;
constexpr std::size_t kNameLengthOffset = 40;
constexpr std::size_t kNameOffset = 44;
constexpr std::size_t kMaxTableNameLength = 200;

// read_u32 — one little-endian 32-bit field out of a page buffer.
// Parameters: buffer — the page bytes; offset — the field's offset.
// Returns: the field's value.
std::uint32_t read_u32(const char* buffer, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, buffer + offset, sizeof(value));
    return value;
}

// write_u32 — one little-endian 32-bit field into a page buffer.
// Parameters: buffer — the page bytes; offset — the field's offset;
// value — the value to store. Returns: nothing.
void write_u32(char* buffer, std::size_t offset, std::uint32_t value) {
    std::memcpy(buffer + offset, &value, sizeof(value));
}

}  // namespace
// BPlusTree::BPlusTree — binds the index to a buffer manager, a table's
// schema, and the one column it orders, deriving the key codec from the
// column's type and width.
//
// Parameters:
//   manager: the session's buffer manager; must outlive the index.
//   schema: the table's schema; must outlive the index (the header
//     stores its name, and create/open re-check the column).
//   column_index: the ordered column's position in the schema.
//
// Returns: nothing; an out-of-range column leaves the codec invalid and
// create/open refuse by name.
BPlusTree::BPlusTree(bufman::BufferManager& manager,
                     const schema::TableSchema& schema,
                     std::size_t column_index)
    : manager_(manager), schema_(schema), column_index_(column_index) {
    if (column_index < schema.column_count()) {
        const schema::ColumnSchema& column = schema.column_info(column_index);
        const std::size_t string_width =
            column.column_type() == tuple::ColumnType::String
                ? column.column_size()
                : 0;
        codec_ = isam::KeyCodec(column.column_type(), string_width);
    }
}

// --- header ------------------------------------------------------------------
// BPlusTree::write_header — persists the in-memory header state (magic,
// key identification, root, first leaf, levels, last block, free-list
// head, entry count, table and column names) into block 0.
//
// Parameters:
//   error: cleared, then set on failure with the pin or name-length
//     cause.
//
// Returns: true on success (block 0 dirty in the pool); false with a
// named error when the block cannot be pinned or the table name exceeds
// the header's capacity.
bool BPlusTree::write_header(std::string& error) {
    const std::string& name = schema_.table_name();
    if (name.size() > kMaxTableNameLength) {
        error = "table name is " + std::to_string(name.size()) +
                " bytes; at most " + std::to_string(kMaxTableNameLength) +
                " fit the header page";
        return false;
    }
    std::size_t frame = 0;
    if (!manager_.pin(file_, 0, frame, error)) {
        return false;
    }
    char* buffer = manager_.frame(frame).buffer;
    write_u32(buffer, kMagicOffset, kBptMagic);
    write_u32(buffer, kKeyTypeOffset,
              static_cast<std::uint32_t>(codec_.key_type()));
    write_u32(buffer, kKeyWidthOffset,
              static_cast<std::uint32_t>(codec_.width()));
    write_u32(buffer, kRootOffset, root_);
    write_u32(buffer, kFirstLeafOffset, first_leaf_);
    write_u32(buffer, kLevelsOffset, levels_);
    write_u32(buffer, kLastBlockOffset, last_block_);
    write_u32(buffer, kFreedHeadOffset, freed_head_);
    const std::uint64_t count = entry_count_;
    std::memcpy(buffer + kEntryCountOffset, &count, 8);
    write_u32(buffer, kNameLengthOffset,
              static_cast<std::uint32_t>(name.size()));
    std::memcpy(buffer + kNameOffset, name.data(), name.size());
    const std::size_t column_offset = kNameOffset + name.size();
    const std::string column_name =
        schema_.column_info(column_index_).column_name();
    write_u32(buffer, column_offset,
              static_cast<std::uint32_t>(column_name.size()));
    std::memcpy(buffer + column_offset + 4, column_name.data(),
                column_name.size());
    release(manager_, frame, true);
    return true;
}
// BPlusTree::read_header — reads block 0's fields into `header` for
// open()'s validation. The kind check happens at the call site; this
// only refuses lengths that would walk off the block.
//
// Parameters:
//   header: the struct to fill.
//   error: cleared, then set on failure.
//
// Returns: true on success; false with a named error when block 0
// cannot be pinned or a name length is impossible ("not a b+tree index
// file").
bool BPlusTree::read_header(Header& header, std::string& error) {
    std::size_t frame = 0;
    if (!manager_.pin(file_, 0, frame, error)) {
        return false;
    }
    const char* buffer = manager_.frame(frame).buffer;
    header.magic = read_u32(buffer, kMagicOffset);
    header.key_type = read_u32(buffer, kKeyTypeOffset);
    header.key_width = read_u32(buffer, kKeyWidthOffset);
    header.root = read_u32(buffer, kRootOffset);
    header.first_leaf = read_u32(buffer, kFirstLeafOffset);
    header.levels = read_u32(buffer, kLevelsOffset);
    header.last_block = read_u32(buffer, kLastBlockOffset);
    header.freed_head = read_u32(buffer, kFreedHeadOffset);
    std::memcpy(&header.entry_count, buffer + kEntryCountOffset, 8);
    const std::uint32_t name_length = read_u32(buffer, kNameLengthOffset);
    const std::size_t column_offset = kNameOffset + name_length;
    const std::uint32_t column_length = read_u32(buffer, column_offset);
    if (name_length > kMaxTableNameLength ||
        column_length > kMaxTableNameLength) {
        error = "not a b+tree index file";
        release(manager_, frame, false);
        return false;
    }
    header.table_name.assign(buffer + kNameOffset, name_length);
    header.column_name.assign(buffer + column_offset + 4, column_length);
    release(manager_, frame, false);
    return true;
}

// --- create / open / close ---------------------------------------------------
// BPlusTree::create — creates a fresh, empty index file at `path` (a
// single header block, root kNoPage, level count 1) and leaves it open.
// The first insert grows the first leaf.
//
// Parameters:
//   path: the file to create; an existing non-empty file is a refusal,
//     an existing empty file is refused too (unlike ISAM — the B+tree
//     keeps one bootstrap path).
//   error: cleared, then set on failure naming the cause.
//
// Returns: true on success (open, header written); false with a named
// error when already open, the codec is invalid, or the file is not
// virgin.
bool BPlusTree::create(const std::string& path, std::string& error) {
    if (open_) {
        error = "cannot create \"" + path + "\": this index is already open";
        return false;
    }
    if (codec_.width() == 0) {
        error = "cannot create \"" + path +
                "\": the column has no record-able key width";
        return false;
    }
    if (!manager_.open_file(path, file_, error)) {
        return false;
    }
    const std::uint64_t blocks = bufman::block_count(file_, error);
    if (!error.empty()) {
        std::string close_error;
        manager_.close_file(file_, close_error);
        return false;
    }
    if (blocks > 0) {
        error = "cannot create \"" + path + "\": the file already holds " +
                std::to_string(blocks) + " block(s)";
        std::string close_error;
        manager_.close_file(file_, close_error);
        return false;
    }
    root_ = idx::kNoPage;
    first_leaf_ = idx::kNoPage;
    levels_ = 1;
    entry_count_ = 0;
    std::size_t frame = 0;
    if (!manager_.alloc_page(file_, 0, frame, error)) {
        std::string close_error;
        manager_.close_file(file_, close_error);
        return false;
    }
    last_block_ = 0;  // the header block exists from here on
    freed_head_ = idx::kNoPage;
    path_ = path;
    const bool wrote = write_header(error);
    release(manager_, frame, wrote);
    if (!wrote) {
        std::string close_error;
        manager_.close_file(file_, close_error);
        return false;
    }
    open_ = true;
    return true;
}
// BPlusTree::open — opens an existing index file, validating the kind
// first (a foreign file — an ISAM index, a heap file — is refused by
// name), then the key type and width, then the table and column names
// against the bound schema. Loads the header state into memory.
//
// Parameters:
//   path: the file to open.
//   error: cleared, then set on failure naming the mismatch.
//
// Returns: true on success (open, state loaded); false with a named
// error on any mismatch ("not a b+tree index file", key/type width,
// "index file is for <table>.<column>").
bool BPlusTree::open(const std::string& path, std::string& error) {
    if (open_) {
        error = "cannot open \"" + path + "\": this index is already open";
        return false;
    }
    if (!manager_.open_file(path, file_, error)) {
        return false;
    }
    Header header;
    if (!read_header(header, error)) {
        std::string close_error;
        manager_.close_file(file_, close_error);
        return false;
    }
    // Kind first: a foreign file (an ISAM index, a heap file) is refused
    // by name, then the identification fields.
    if (header.magic != kBptMagic) {
        error = "not a b+tree index file";
        std::string close_error;
        manager_.close_file(file_, close_error);
        return false;
    }
    if (header.key_type != static_cast<std::uint32_t>(codec_.key_type()) ||
        header.key_width != codec_.width()) {
        error = "index file holds " + std::to_string(header.key_width) +
                "-byte keys of another type; the column needs " +
                std::to_string(codec_.width());
        std::string close_error;
        manager_.close_file(file_, close_error);
        return false;
    }
    if (header.table_name != schema_.table_name() ||
        header.column_name !=
            schema_.column_info(column_index_).column_name()) {
        error = "index file is for " + header.table_name + "." +
                header.column_name;
        std::string close_error;
        manager_.close_file(file_, close_error);
        return false;
    }
    root_ = header.root;
    first_leaf_ = header.first_leaf;
    levels_ = header.levels;
    last_block_ = header.last_block;
    freed_head_ = header.freed_head;
    entry_count_ = static_cast<std::size_t>(header.entry_count);
    path_ = path;
    open_ = true;
    return true;
}
// BPlusTree::close — flushes and closes this index's file through the
// shared buffer manager (the dirty pages reach disk here).
//
// Parameters:
//   error: cleared, then set on failure with the manager's cause.
//
// Returns: true on success (is_open() becomes false); false with a
// named error when the index is not open or the file cannot close.
bool BPlusTree::close(std::string& error) {
    if (!open_) {
        error = "index is not open";
        return false;
    }
    if (!manager_.close_file(file_, error)) {
        return false;
    }
    open_ = false;
    return true;
}

// --- pin and allocation helpers -----------------------------------------------
// BPlusTree::pin_leaf — pins a block through the manager and binds a
// leaf view over it, validating on attach. Unpins on failure, so a
// refused pin never leaks.
//
// Parameters:
//   block: the block id to pin.
//   page: the view to bind over the frame's buffer.
//   frame: receives the pin's frame index (the caller releases it).
//   error: cleared, then set on failure.
//
// Returns: true on success (the caller owns the pin); false with a
// named error when the pin or the attach fails.
bool BPlusTree::pin_leaf(std::uint32_t block, BptLeafPage& page,
                         std::size_t& frame, std::string& error) {
    if (!manager_.pin(file_, block, frame, error)) {
        return false;
    }
    page.bind(manager_.frame(frame).buffer, codec_.width());
    if (!page.attach(error)) {
        release(manager_, frame, false);
        return false;
    }
    return true;
}
// BPlusTree::pin_separator — pins a block and binds a separator view
// over it, validating on attach; unpins on failure.
//
// Parameters:
//   block: the block id to pin.
//   page: the view to bind over the frame's buffer.
//   frame: receives the pin's frame index.
//   error: cleared, then set on failure.
//
// Returns: true on success (the caller owns the pin); false with a
// named error when the pin or the attach fails.
bool BPlusTree::pin_separator(std::uint32_t block, BptSeparatorPage& page,
                              std::size_t& frame, std::string& error) {
    if (!manager_.pin(file_, block, frame, error)) {
        return false;
    }
    page.bind(manager_.frame(frame).buffer, codec_.width());
    if (!page.attach(error)) {
        release(manager_, frame, false);
        return false;
    }
    return true;
}
// BPlusTree::alloc_next_block — claims the next data block: a block
// freed by a merge is reused first (its own first word chained the free
// list), otherwise one past `last_block_` is allocated fresh. The
// caller formats the page and releases the frame dirty.
//
// Parameters:
//   block: receives the claimed block id.
//   frame: receives the pinned frame index (the caller releases it).
//   error: cleared, then set on failure.
//
// Returns: true on success (the caller owns the pin); false with a
// named error when neither the free list nor the allocator can serve.
bool BPlusTree::alloc_next_block(std::uint32_t& block, std::size_t& frame,
                                 std::string& error) {
    // A block freed by a merge is reused before anything new is
    // allocated; its own first word chained the free list.
    if (freed_head_ != idx::kNoPage) {
        block = freed_head_;
        if (!manager_.pin(file_, block, frame, error)) {
            return false;
        }
        std::uint32_t next = idx::kNoPage;
        std::memcpy(&next, manager_.frame(frame).buffer, 4);
        freed_head_ = next;
        return true;  // the caller formats the page and releases it dirty
    }
    block = last_block_ == idx::kNoPage ? 1 : last_block_ + 1;
    if (!manager_.alloc_page(file_, block, frame, error)) {
        return false;
    }
    last_block_ = block;
    return true;
}
// BPlusTree::chain_freed — pushes a merged-away block onto the free
// list: the block's own first word is rewritten to the current head and
// the head moves to this block. The caller holds the block pinned
// through the surgery and releases it dirty.
//
// Parameters:
//   block: the vanishing block's id.
//   buffer: the pinned frame's memory (the chain link is written in
//     place).
//
// Returns: nothing; freed_head_ advances to `block`.
void BPlusTree::chain_freed(std::uint32_t block, char* buffer) {
    write_u32(buffer, 0, freed_head_);
    freed_head_ = block;
}
// BPlusTree::min_leaf_entries — the order-d leaf minimum: half the leaf
// capacity, rounded down (design D3; a full node holds exactly 2d, so
// every split boundary is a named position). The root is exempt.
//
// Parameters: none.
//
// Returns: d for leaves.
std::size_t BPlusTree::min_leaf_entries() const {
    return bptree::leaf_capacity(codec_.width()) / 2;
}
// BPlusTree::max_leaf_entries — the order-d leaf maximum: exactly 2d
// (capacity rounded down to even), so a full leaf plus one arriving
// entry splits at the named boundaries (slide 35).
//
// Parameters: none.
//
// Returns: 2d for leaves.
std::size_t BPlusTree::max_leaf_entries() const {
    return (bptree::leaf_capacity(codec_.width()) / 2) * 2;
}
// BptSeparatorPage::min_separator_keys — the order-d separator minimum:
// half the separator capacity, rounded down. The root is exempt.
//
// Parameters: none.
//
// Returns: d for separator pages.
std::size_t BPlusTree::min_separator_keys() const {
    return bptree::separator_capacity(codec_.width()) / 2;
}
// BPlusTree::max_separator_keys — the order-d separator maximum:
// exactly 2d pairs, so a full separator page plus one arriving pair
// splits with the middle key moving up (slide 37).
//
// Parameters: none.
//
// Returns: 2d for separator pages.
std::size_t BPlusTree::max_separator_keys() const {
    return (bptree::separator_capacity(codec_.width()) / 2) * 2;
}

// --- build (design D7) ---------------------------------------------------------
// BPlusTree::build — bulk build (design D7): one in-memory sort, then
// leaves sized evenly (every page within d..2d by construction) filling
// left to right and linking the doubly linked chain, then separator
// levels built bottom-up from each page's first key until one page —
// the root — holds a level. An empty entry list is a valid empty index.
//
// Parameters:
//   path: the file to create and fill (must not exist non-empty).
//   entries: the (value, rid) pairs; unsorted input is fine.
//   error: cleared, then set on failure.
//
// Returns: true on success (open, entry_count_ = the pair count); false
// with a named error on any refusal — the file is left as create() made
// it.
bool BPlusTree::build(
    const std::string& path,
    const std::vector<std::pair<tuple::Column, idx::Rid>>& entries,
    std::string& error) {
    if (open_) {
        error = "cannot build \"" + path + "\": this index is already open";
        return false;
    }
    if (!create(path, error)) {
        return false;
    }
    if (entries.empty()) {
        // A valid empty index: the first insert grows its first leaf.
        return true;
    }

    // One in-memory sort orders the pairs; the codec's byte order IS the
    // value order.
    std::vector<idx::DataEntry> sorted;
    sorted.reserve(entries.size());
    for (const auto& [value, rid] : entries) {
        idx::DataEntry entry;
        entry.key.resize(codec_.width());
        if (!codec_.encode(value, entry.key.data(), error)) {
            return false;
        }
        entry.rid = rid;
        sorted.push_back(std::move(entry));
    }
    std::sort(sorted.begin(), sorted.end(),
              [this](const idx::DataEntry& a, const idx::DataEntry& b) {
                  return codec_.compare(a.key.data(), b.key.data()) < 0;
              });

    // Leaves fill left to right linking the doubly linked chain, sized
    // so every leaf stays within the order rule: p = ceil(N / 2d) pages
    // of base (or base+1) entries, each between d and 2d. Each leaf's
    // first key feeds the first separator level.
    const std::size_t leaf_fill = max_leaf_entries();
    const std::size_t leaf_pages =
        (sorted.size() + leaf_fill - 1) / leaf_fill;
    const std::size_t leaf_base = sorted.size() / leaf_pages;
    const std::size_t leaf_extra = sorted.size() % leaf_pages;
    std::vector<std::pair<std::string, std::uint32_t>> level;  // first key, block
    {
        std::uint32_t prev_block = idx::kNoPage;
        std::size_t index = 0;
        for (std::size_t p = 0; p < leaf_pages; ++p) {
            const std::size_t fill = leaf_base + (p < leaf_extra ? 1 : 0);
            std::uint32_t block = 0;
            std::size_t frame = 0;
            if (!alloc_next_block(block, frame, error)) {
                return false;
            }
            BptLeafPage leaf;
            leaf.bind(manager_.frame(frame).buffer, codec_.width());
            const bool more_leaves = p + 1 < leaf_pages;
            leaf.format(prev_block,
                        more_leaves ? block + 1 : idx::kNoPage);
            for (std::size_t i = 0; i < fill; ++i) {
                if (!leaf.insert_entry(sorted[index + i], codec_, error)) {
                    release(manager_, frame, false);
                    return false;
                }
            }
            release(manager_, frame, true);
            level.emplace_back(sorted[index].key, block);
            prev_block = block;
            index += fill;
        }
        first_leaf_ = level.front().second;
    }

    // Separator levels build bottom-up: each level holds (first key,
    // block) for every page of the level below except the leftmost,
    // grouped into pages sized the same even way (each between d and 2d
    // pairs), until one page holds a level — the root.
    std::uint32_t separator_levels = 0;
    while (level.size() > 1) {
        const std::size_t sep_fill = max_separator_keys();
        const std::size_t sep_pages = (level.size() + sep_fill - 1) / sep_fill;
        const std::size_t sep_base = level.size() / sep_pages;
        const std::size_t sep_extra = level.size() % sep_pages;
        std::vector<std::pair<std::string, std::uint32_t>> next_level;
        std::size_t position = 0;
        for (std::size_t p = 0; p < sep_pages; ++p) {
            const std::size_t group = sep_base + (p < sep_extra ? 1 : 0);
            std::uint32_t block = 0;
            std::size_t frame = 0;
            if (!alloc_next_block(block, frame, error)) {
                return false;
            }
            BptSeparatorPage page;
            page.bind(manager_.frame(frame).buffer, codec_.width());
            page.format();
            // The group's first page is this page's first child; every
            // later page in the group contributes (first key, block).
            page.set_first_child(level[position].second);
            for (std::size_t i = position + 1; i < position + group; ++i) {
                if (!page.append(
                        idx::IndexEntry{level[i].first, level[i].second},
                        error)) {
                    release(manager_, frame, false);
                    return false;
                }
            }
            release(manager_, frame, true);
            next_level.emplace_back(level[position].first, block);
            position += group;
        }
        level = std::move(next_level);
        ++separator_levels;
    }
    root_ = level.front().second;
    levels_ = 1 + separator_levels;
    entry_count_ = sorted.size();
    return write_header(error);
}

// --- the recursion contract: search --------------------------------------------
// BPlusTree::tree_search — the lecture's treeSearch (slides 17, 24):
// pin one separator, take the child after the LAST separator <= the
// key, recurse; `levels_remaining == 1` means `block` IS the landing
// leaf. The lecture's `Ki <= key` routing, literally.
//
// Parameters:
//   block: the page to route from (the root on the first call).
//   levels_remaining: the levels from here down, the leaf level itself
//     included.
//   key_bytes: the encoded target key, exactly codec.width() bytes.
//   error: cleared, then set on failure.
//
// Returns: the landing leaf's block id (kNoPage for an empty tree or on
// a pin failure, with the error named).
std::uint32_t BPlusTree::tree_search(std::uint32_t block,
                                     std::uint32_t levels_remaining,
                                     const std::string& key_bytes,
                                     std::string& error) {
    // TODO(student): implement the lecture's treeSearch (slides 17, 24).
    // Pin this page's separator view; when levels_remaining is 1 this
    // page IS the landing leaf — return its block. Otherwise take the
    // child after the LAST separator <= key_bytes
    // (routing_child_index), release the pin, and recurse into that
    // child with levels_remaining - 1. An empty tree (kNoPage) also
    // returns kNoPage.
    
    (void)block;
    (void)levels_remaining;
    (void)key_bytes;
    (void)error;
    return idx::kNoPage;
}
// BPlusTree::run_start — the equal-key run's leftmost leaf: walks left
// from the landing leaf while the previous leaf's LAST key EQUALS the
// target — the exact boundary test that gathers the boundary duplicates
// the copy-up convention can leave in the left sibling's tail (design
// D4). Leaves are globally ordered, so equality is precise: a >= test
// would over-walk.
//
// Parameters:
//   landing: the tree_search landing leaf.
//   key_bytes: the encoded target key.
//   error: cleared, then set on failure.
//
// Returns: the run's first leaf's block id (the landing itself when no
// left neighbour's tail equals the target); kNoPage on a pin failure.
std::uint32_t BPlusTree::run_start(std::uint32_t landing,
                                   const std::string& key_bytes,
                                   std::string& error) {
    // TODO(student): the equal-key run's leftmost leaf (design D4).
    // Walk left from `landing` while the PREVIOUS leaf's LAST key equals
    // the target (exact equality — a >= test over-walks): pin the
    // current leaf, read its prev link, release; pin the previous leaf,
    // compare its last entry's key with key_bytes, release; stop and
    // return `current` when prev is kNoPage or the tail does not equal
    // the target, else continue from prev.
    (void)landing;
    (void)key_bytes;
    (void)error;
    return idx::kNoPage;
}
// BPlusTree::gather_run — collects every entry whose key equals the
// target, scanning leaves left to right from `start`: the run continues
// while a leaf's last key still compares equal. Lookup's answer walk.
//
// Parameters:
//   key_bytes: the encoded target key.
//   start: the run's leftmost leaf (run_start's result).
//   out: cleared by the caller; receives the rids in leaf order.
//   error: cleared, then set on failure.
//
// Returns: true on success (out holds the answer, possibly empty);
// false with a named error on a pin failure.
bool BPlusTree::gather_run(const std::string& key_bytes, std::uint32_t start,
                           std::vector<idx::Rid>& out, std::string& error) {
    // TODO(student): collect the equal-key entries leaf by leaf. For
    // each leaf from `start`: lower_bound(key_bytes), then scan forward
    // while keys compare equal, pushing each entry's rid into `out`;
    // the run continues into next_leaf while the leaf's LAST key equals
    // the target. Stop at the first leaf whose last key differs.
    (void)key_bytes;
    (void)start;
    (void)out;
    (void)error;
    return false;
}
// BPlusTree::locate_in_run — finds the exact (key, rid) entry within
// the equal-key run: the leaves are scanned for the pair, so the erase
// target can sit left of the landing leaf (boundary duplicates) or
// right (a run spanning leaves).
//
// Parameters:
//   key_bytes: the encoded target key.
//   rid: the exact row address to find.
//   start: the run's leftmost leaf.
//   leaf_block: receives the holding leaf's block id on success.
//   pos: receives the entry position within that leaf.
//   error: cleared, then set on failure.
//
// Returns: true when the exact pair is found (the two out-parameters
// name it); false with a named error ("no index entry for key ... and
// rid ...") when the run does not hold it.
bool BPlusTree::locate_in_run(const std::string& key_bytes, idx::Rid rid,
                              std::uint32_t start, std::uint32_t& leaf_block,
                              std::size_t& pos, std::string& error) {
    // TODO(student): find the exact (key_bytes, rid) pair within the
    // run: per leaf, lower_bound then scan the equal-key entries for the
    // matching rid — report that leaf and position through the
    // out-parameters. Continue into next_leaf while the last key equals
    // the target; a run that exhausts without the rid is a named error
    // ("no index entry for key ... and rid ...").
    (void)key_bytes;
    (void)rid;
    (void)start;
    (void)leaf_block;
    (void)pos;
    (void)error;
    return false;
}
// BPlusTree::lookup — all entries whose key equals `key`: descend by
// the lecture's routing, step left to the run's start, gather forward.
// Duplicates are contiguous by key; an empty index answers nothing.
//
// Parameters:
//   key: the value to encode and search for.
//   out: cleared, then filled with the matching rids.
//   error: cleared, then set on failure.
//
// Returns: true on success — including "no matches" (out empty); false
// with a named error when the index is closed, the value cannot be
// encoded, or a pin fails.
bool BPlusTree::lookup(const tuple::Column& key, std::vector<idx::Rid>& out,
                       std::string& error) {
    out.clear();
    if (!open_) {
        error = "index is not open";
        return false;
    }
    if (root_ == idx::kNoPage) {
        return true;  // an empty index answers nothing
    }
    std::string key_bytes(codec_.width(), '\0');
    if (!codec_.encode(key, key_bytes.data(), error)) {
        return false;
    }
    const std::uint32_t landing =
        tree_search(root_, levels_, key_bytes, error);
    if (!error.empty()) {
        return false;
    }
    const std::uint32_t start = run_start(landing, key_bytes, error);
    if (!error.empty()) {
        return false;
    }
    return gather_run(key_bytes, start, out, error);
}
// BPlusTree::range — entries with lo <= key <= hi (bounds optional,
// inclusivity per bound), in ascending key order with duplicates
// adjacent: the walk starts at the lower bound's run start (or the
// chain's first leaf) and stops at the first key past the upper bound.
//
// Parameters:
//   lo: the optional lower bound (value + inclusive).
//   hi: the optional upper bound (value + inclusive).
//   out: cleared, then filled with the matching rids.
//   error: cleared, then set on failure.
//
// Returns: true on success — including an empty window; false with a
// named error when the index is closed, a bound cannot be encoded, or a
// pin fails.
bool BPlusTree::range(const std::optional<RangeBound>& lo,
                      const std::optional<RangeBound>& hi,
                      std::vector<idx::Rid>& out, std::string& error) {
    out.clear();
    if (!open_) {
        error = "index is not open";
        return false;
    }
    if (root_ == idx::kNoPage) {
        return true;  // an empty index answers nothing
    }
    std::string lo_bytes;
    bool have_lo = false;
    if (lo) {
        lo_bytes.resize(codec_.width());
        if (!codec_.encode(lo->value, lo_bytes.data(), error)) {
            return false;
        }
        have_lo = true;
    }
    std::string hi_bytes;
    if (hi) {
        hi_bytes.resize(codec_.width());
        if (!codec_.encode(hi->value, hi_bytes.data(), error)) {
            return false;
        }
    }
    std::uint32_t start = first_leaf_;
    if (have_lo) {
        const std::uint32_t landing =
            tree_search(root_, levels_, lo_bytes, error);
        if (!error.empty()) {
            return false;
        }
        start = run_start(landing, lo_bytes, error);
        if (!error.empty()) {
            return false;
        }
    }
    // Walk the chain forward from `start`, emitting the entries both
    // bounds accept; the first key past the upper bound ends the walk.
    std::uint32_t current = start;
    while (current != idx::kNoPage) {
        std::size_t frame = 0;
        BptLeafPage leaf;
        if (!pin_leaf(current, leaf, frame, error)) {
            return false;
        }
        const std::size_t n = leaf.count();
        const std::uint32_t next = leaf.next_leaf();
        bool past_upper = false;
        for (std::size_t i = 0; i < n; ++i) {
            const idx::DataEntry entry = leaf.get_entry(i);
            if (have_lo) {
                const int cmp =
                    codec_.compare(entry.key.data(), lo_bytes.data());
                if (cmp < 0 || (cmp == 0 && !lo->inclusive)) {
                    continue;
                }
            }
            if (hi) {
                const int cmp =
                    codec_.compare(entry.key.data(), hi_bytes.data());
                if (cmp > 0 || (cmp == 0 && !hi->inclusive)) {
                    past_upper = true;
                    break;
                }
            }
            out.push_back(entry.rid);
        }
        release(manager_, frame, false);
        if (past_upper) {
            break;
        }
        current = next;
    }
    return true;
}

// --- the recursion contract: insert --------------------------------------------
// BPlusTree::insert — the lecture's insert (slide 34) as a wrapper: an
// empty index grows its first leaf; otherwise insert_aux descends and
// the wrapper allocates a new root when the old root split (the height
// grows by exactly one). entry_count_ and the header follow.
//
// Parameters:
//   key: the value to encode and insert.
//   rid: the row address the entry maps to.
//   error: cleared, then set on failure.
//
// Returns: true on success (the entry is findable by lookup and range
// immediately); false with a named error when the index is closed, the
// value cannot be encoded, or any pin/alloc fails.
bool BPlusTree::insert(const tuple::Column& key, idx::Rid rid,
                       std::string& error) {
    if (!open_) {
        error = "index is not open";
        return false;
    }
    std::string key_bytes(codec_.width(), '\0');
    if (!codec_.encode(key, key_bytes.data(), error)) {
        return false;
    }

    // An empty index grows its first leaf; it is simultaneously the root
    // (levels_ == 1).
    if (root_ == idx::kNoPage) {
        std::uint32_t block = 0;
        std::size_t frame = 0;
        if (!alloc_next_block(block, frame, error)) {
            return false;
        }
        BptLeafPage leaf;
        leaf.bind(manager_.frame(frame).buffer, codec_.width());
        leaf.format(idx::kNoPage, idx::kNoPage);
        if (!leaf.insert_entry(idx::DataEntry{key_bytes, rid}, codec_,
                               error)) {
            release(manager_, frame, false);
            return false;
        }
        release(manager_, frame, true);
        root_ = block;
        first_leaf_ = block;
        levels_ = 1;
        entry_count_ = 1;
        return write_header(error);
    }

    Split split;
    if (!insert_aux(root_, levels_, key_bytes, rid, split, error)) {
        return false;
    }
    if (split.new_block != idx::kNoPage) {
        // The old root split: a new root above both halves (slide 34).
        std::uint32_t block = 0;
        std::size_t frame = 0;
        if (!alloc_next_block(block, frame, error)) {
            return false;
        }
        BptSeparatorPage new_root;
        new_root.bind(manager_.frame(frame).buffer, codec_.width());
        new_root.format();
        new_root.set_first_child(root_);
        if (!new_root.append(
                idx::IndexEntry{split.separator_key, split.new_block},
                error)) {
            release(manager_, frame, false);
            return false;
        }
        release(manager_, frame, true);
        root_ = block;
        levels_ += 1;
    }
    entry_count_ += 1;
    return write_header(error);
}
// BPlusTree::insert_aux — the lecture's insertAux (slides 34-37): at
// the leaf, insert or split (keep d, give d+1, copy the new leaf's
// smallest key up); at a separator, route, recurse, then seat the
// carried (separator, new child) pair after the descended child —
// splitting this page too when it is full (the middle key moves up).
// The Split carrier reports the new page and separator to the parent.
//
// Parameters:
//   block: the page to act on (the root on the first call).
//   levels_remaining: the levels from here down, the leaf level
//     included.
//   key_bytes: the encoded key to insert.
//   rid: the row address.
//   split: the upward carrier — zeroed here; filled when this page
//     split (separator_key routes to new_block).
//   error: cleared, then set on failure.
//
// Returns: true on success; false with a named error on any pin, alloc,
// or insert failure.
bool BPlusTree::insert_aux(std::uint32_t block, std::uint32_t levels_remaining,
                           const std::string& key_bytes, idx::Rid rid,
                           Split& split, std::string& error) {
    // TODO(student): the lecture's insertAux (slides 34-37). Zero
    // `split`. At a leaf (levels_remaining <= 1) delegate to
    // insert_leaf. At a separator: pin, route (routing_child_index),
    // release, recurse into the child; when the child split, re-pin and
    // seat (child_split.separator_key, child_split.new_block) after the
    // descended child (insert_pair_after) — or, when this page is full
    // (2d pairs), split per slide 37: the middle pair's key moves up and
    // its child becomes the new page's first child
    // (move_tail_to(middle + 1)), filling `split` for the parent. The
    // wrapper turns a root split into a new root.
    (void)block;
    (void)levels_remaining;
    (void)key_bytes;
    (void)rid;
    (void)split;
    (void)error;
    return false;
}
// BPlusTree::insert_leaf — the leaf phase of insert_aux: room within
// 2d means a plain sorted insert; a full leaf (2d entries) splits per
// slide 35 — this page keeps d, the new leaf takes d+1, the separator
// copied up is the new leaf's smallest key (copy-up), and the chain
// links both ways. The arriving entry then takes its place in the
// correct half (equal keys go right, where the routing sends their
// duplicates).
//
// Parameters:
//   block: the landing leaf's block id.
//   key_bytes: the encoded key to insert.
//   rid: the row address.
//   split: the upward carrier; filled when the leaf split.
//   error: cleared, then set on failure.
//
// Returns: true on success; false with a named error on any pin, alloc,
// or insert failure.
bool BPlusTree::insert_leaf(std::uint32_t block, const std::string& key_bytes,
                            idx::Rid rid, Split& split, std::string& error) {
    // TODO(student): the leaf phase of insertAux (slide 35). Room within
    // 2d (max_leaf_entries): plain sorted insert_entry, release dirty,
    // no split. Full (2d entries): release, allocate a new block
    // (alloc_next_block), re-pin, move_tail_to(keep) with
    // keep = min_leaf_entries, link the chain both ways
    // (fresh.set_prev_leaf(block); fresh.set_next_leaf(old_next);
    // leaf.set_next_leaf(new_block); the old next leaf's prev becomes
    // new_block), then insert the arriving entry into the correct half
    // (equal keys go RIGHT — where the routing sends their duplicates).
    // Fill `split` with the copied-up separator (the new leaf's smallest
    // key) and the new block.
    (void)block;
    (void)key_bytes;
    (void)rid;
    (void)split;
    (void)error;
    return false;
}

// --- the recursion contract: erase ----------------------------------------------
// BPlusTree::erase — removes the one entry equal to (key, rid): locate
// first (the landing leaf plus the equal-key run — a miss is a named
// error with nothing changed), then either the recursive path (the leaf
// erases, the parents rebalance, a single-child root is dropped) or the
// crossed-subtree path (erase in place; the leaf's own state decides
// whether the rebuild repair runs). See the design's trade-off.
//
// Parameters:
//   key: the value to encode and erase.
//   rid: the exact row address to remove.
//   error: cleared, then set on failure.
//
// Returns: true on success (lookups and ranges no longer return the
// rid; the count dropped by one; the tree verifies clean); false with a
// named error when the index is closed or the pair is absent.
bool BPlusTree::erase(const tuple::Column& key, idx::Rid rid,
                      std::string& error) {
    if (!open_) {
        error = "index is not open";
        return false;
    }
    if (root_ == idx::kNoPage) {
        error = "index is empty; no entry to erase";
        return false;
    }
    std::string key_bytes(codec_.width(), '\0');
    if (!codec_.encode(key, key_bytes.data(), error)) {
        return false;
    }

    // Locate the exact entry first: the landing leaf plus the run (a
    // miss is a named error with nothing changed).
    const std::uint32_t landing =
        tree_search(root_, levels_, key_bytes, error);
    if (!error.empty()) {
        return false;
    }
    const std::uint32_t start = run_start(landing, key_bytes, error);
    if (!error.empty()) {
        return false;
    }
    std::uint32_t target_leaf = idx::kNoPage;
    std::size_t pos = 0;
    if (!locate_in_run(key_bytes, rid, start, target_leaf, pos, error)) {
        return false;
    }

    if (target_leaf == landing) {
        // The recursive path: the leaf erases, the parents rebalance.
        EraseUp up;
        if (!erase_rec(root_, levels_, key_bytes, rid, up, error)) {
            return false;
        }
        // A root left with a single child is dropped (slides 48-49).
        while (levels_ > 1) {
            std::size_t frame = 0;
            BptSeparatorPage root_page;
            if (!pin_separator(root_, root_page, frame, error)) {
                return false;
            }
            if (root_page.count() != 0) {
                release(manager_, frame, false);
                break;
            }
            const std::uint32_t child = root_page.first_child();
            chain_freed(root_, manager_.frame(frame).buffer);
            release(manager_, frame, true);
            root_ = child;
            levels_ -= 1;
        }
    } else {
        // The equal-key run crosses a subtree boundary: erase in place,
        // then let the leaf's own state decide the repair.
        std::size_t frame = 0;
        BptLeafPage leaf;
        if (!pin_leaf(target_leaf, leaf, frame, error)) {
            return false;
        }
        const bool was_first = pos == 0;
        if (!leaf.erase_entry(pos, error)) {
            release(manager_, frame, false);
            return false;
        }
        const bool underflow = leaf.count() < min_leaf_entries();
        const bool first_changed = was_first && leaf.count() > 0;
        release(manager_, frame, true);
        if (underflow || first_changed) {
            // The simple correct repair for the rare crossed case:
            // rebuild from the leaf chain (design trade-off).
            if (!rebuild_from_leaves(error)) {
                return false;
            }
        }
    }
    entry_count_ -= 1;
    return write_header(error);
}
// BPlusTree::erase_rec — the erase cascade (slides 38-49) as a carrier
// recursion: at the leaf, remove the exact entry and report underflow
// and first-key changes; at a separator, route, recurse, then repair
// the child (redistribute or merge, a first-key separator rewrite).
// The EraseUp carrier reports this page's own state to its parent.
//
// Parameters:
//   block: the page to act on (the root on the first call).
//   levels_remaining: the levels from here down, the leaf level
//     included.
//   key_bytes: the encoded key of the entry being erased.
//   rid: the exact row address.
//   up: the upward carrier — zeroed here; filled with this page's
//     underflow and first-key-change state.
//   error: cleared, then set on failure.
//
// Returns: true on success; false with a named error when the exact
// entry is missing (a pre-located entry going missing means real
// corruption) or a pin fails.
bool BPlusTree::erase_rec(std::uint32_t block, std::uint32_t levels_remaining,
                          const std::string& key_bytes, idx::Rid rid,
                          EraseUp& up, std::string& error) {
    // TODO(student): the erase cascade (slides 38-49). Zero `up`. At a
    // leaf (levels_remaining <= 1) delegate to erase_leaf. At a
    // separator: pin, route, release, recurse into the child; when the
    // child underflowed, repair it through fix_child_underflow; when
    // only its first key moved, rewrite this page's separator for that
    // child (set_separator_key) — or propagate the change upward when
    // the child is the first child. Fill `up` with this page's own
    // post-repair state.
    (void)block;
    (void)levels_remaining;
    (void)key_bytes;
    (void)rid;
    (void)up;
    (void)error;
    return false;
}
// BPlusTree::erase_leaf — the leaf phase of erase_rec: find the exact
// (key, rid) entry in this leaf's equal-key run, erase it (the
// tail-shift closes the slot), and report the leaf's new state upward —
// underflow (count < d) and whether the first key moved (the parent's
// separator must follow the subtree minimum).
//
// Parameters:
//   block: the landing leaf's block id.
//   key_bytes: the encoded key.
//   rid: the exact row address.
//   up: the upward carrier; filled with the leaf's new state.
//   error: cleared, then set on failure.
//
// Returns: true on success; false with a named error when the pair is
// not in this leaf ("no index entry for key ... and rid ...").
bool BPlusTree::erase_leaf(std::uint32_t block, const std::string& key_bytes,
                           idx::Rid rid, EraseUp& up, std::string& error) {
    // TODO(student): the leaf phase of the erase. Find the exact
    // (key_bytes, rid) entry: lower_bound, then scan the equal-key run
    // for the rid — a miss is the named "no index entry" error with
    // nothing changed. Erase with erase_entry (the tail-shift), then
    // fill `up`: underflow when count < min_leaf_entries(); when the
    // erased entry was the first, report whether the first key moved
    // (up.first_changed + up.new_first); an emptied leaf reports both.
    (void)block;
    (void)key_bytes;
    (void)rid;
    (void)up;
    (void)error;
    return false;
}
// BPlusTree::fix_child_underflow — repairs a child holding fewer than d
// entries: redistribute with a sibling that can lend (left preferred,
// then right), else merge with a sibling and drop the separator. All
// surgery pins are acquired up front in ascending block order, so the
// mutation is pure memory writes that cannot fail midway.
//
// Parameters:
//   block: the parent page's block id (this page's own underflow is
//     reported upward through `up`).
//   levels_remaining: the levels from here down — the child is a leaf
//     when this is 2.
//   child_index: the child's routing position in this page.
//   up: the upward carrier; filled with this page's post-repair state.
//   error: cleared, then set on failure.
//
// Returns: true on success; false with a named error on any pin
// failure.
bool BPlusTree::fix_child_underflow(std::uint32_t block,
                                    std::uint32_t levels_remaining,
                                    std::size_t child_index, EraseUp& up,
                                    std::string& error) {
    // TODO(student): repair a child holding fewer than d entries
    // (slides 40-49). Read phase: pin this page, take the child's block
    // and its siblings' blocks and the pair count, release. Count the
    // child and both siblings (quick_count). Redistribute when a sibling
    // can lend (left preferred, then right — borrow_from_left /
    // borrow_from_right); otherwise merge_child. Fill `up` with this
    // page's own underflow state afterwards (a merge dropped one pair).
    (void)block;
    (void)levels_remaining;
    (void)child_index;
    (void)up;
    (void)error;
    return false;
}
// BPlusTree::quick_count — a page's occupancy for the repair decision
// (can the sibling lend?), read through a short-lived pin.
//
// Parameters:
//   block: the page to count.
//   leaf: which caste to bind (true = BptLeafPage, false =
//     BptSeparatorPage) — the caller knows it from the level.
//   error: cleared, then set on failure.
//
// Returns: the page's count (0 on failure, with the error named).
std::size_t BPlusTree::quick_count(std::uint32_t block, bool leaf,
                                   std::string& error) {
    std::size_t frame = 0;
    std::size_t count = 0;
    if (leaf) {
        BptLeafPage page;
        if (!pin_leaf(block, page, frame, error)) {
            return 0;
        }
        count = page.count();
    } else {
        BptSeparatorPage page;
        if (!pin_separator(block, page, frame, error)) {
            return 0;
        }
        count = page.count();
    }
    release(manager_, frame, false);
    return count;
}

// Pins the surgery's pages (ascending block order, all before the first
// write) and hands back their frames. Releases everything on failure.
// pin_surgery — the borrow/merge discipline in one place: every page
// the surgery touches is pinned up front in ascending block order (the
// TUP2 list-surgery discipline), so the mutation that follows is pure
// memory writes that cannot fail midway. Parameters: manager — the
// owner; file — the index file; blocks — the pages to pin (deduped and
// sorted here); frames — receives block -> frame; error — cleared, then
// set on failure. Returns: true when every pin is held (release_surgery
// owns them); false with a named error after releasing what it got.
bool pin_surgery(bufman::BufferManager& manager, bufman::File file,
                 const std::vector<std::uint32_t>& blocks,
                 std::unordered_map<std::uint32_t, std::size_t>& frames,
                 std::string& error) {
    std::vector<std::uint32_t> sorted = blocks;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    for (const std::uint32_t b : sorted) {
        std::size_t frame = 0;
        if (!manager.pin(file, b, frame, error)) {
            for (const auto& [pinned, f] : frames) {
                release(manager, f, false);
            }
            return false;
        }
        frames.emplace(b, frame);
    }
    return true;
}

// release_surgery — releases every surgery pin dirty (the mutation
// happened; the pages are written back). Parameters: manager — the
// owner; frames — the block -> frame map from pin_surgery. Returns:
// nothing.
void release_surgery(bufman::BufferManager& manager,
                     std::unordered_map<std::uint32_t, std::size_t>& frames) {
    for (const auto& [block, frame] : frames) {
        release(manager, frame, true);
    }
}
// BPlusTree::borrow_from_left — redistribution from the left sibling
// (slides 40-41 mirrored): a leaf child gains the sibling's last entry
// at its front; a separator child gains the sibling's last child and
// key by the rotation. The separator between the siblings is rewritten
// to stay exact. Surgery pins in ascending block order, first.
//
// Parameters:
//   block: the parent's block id.
//   child_index: the underflowed child's routing position (> 0 — a
//     left borrow needs a left sibling).
//   child_block: the child's block id.
//   left_block: the sibling's block id.
//   child_is_leaf: which caste the surgery moves.
//   child_count: the child's occupancy (read by the caller's decision).
//   up: the upward carrier; cleared of underflow by the repair.
//   error: cleared, then set on failure.
//
// Returns: true on success (the child is back within the order rule);
// false with a named error on a pin failure (nothing written).
bool BPlusTree::borrow_from_left(std::uint32_t block, std::size_t child_index,
                                 std::uint32_t child_block,
                                 std::uint32_t left_block, bool child_is_leaf,
                                 std::size_t child_count, EraseUp& up,
                                 std::string& error) {
    // TODO(student): redistribute from the left sibling (slides 40-41
    // mirrored). Pin parent, sibling, child in ascending block order
    // (pin_surgery). Leaf child: move the sibling's last entry to the
    // child's front and rewrite the separator between them to the moved
    // key. Separator child: rotate — the sibling's last child and last
    // key move through the parent (the child's new first child and first
    // key; the parent's separator becomes the moved key). Release
    // everything dirty; clear up.underflow.
    (void)block;
    (void)child_index;
    (void)child_block;
    (void)left_block;
    (void)child_is_leaf;
    (void)child_count;
    (void)up;
    (void)error;
    return false;
}
// BPlusTree::borrow_from_right — redistribution from the right sibling
// (slide 41): a leaf child gains the sibling's first entry at its end;
// a separator child gains the sibling's first child and key by the
// rotation. The parent's separator becomes the sibling's new minimum —
// "copy up 67 which is min key on remaining child". An emptied child
// also rewrites (or propagates) its LEFT boundary.
//
// Parameters:
//   block: the parent's block id.
//   child_index: the underflowed child's routing position.
//   child_block: the child's block id.
//   right_block: the sibling's block id.
//   child_is_leaf: which caste the surgery moves.
//   child_count: the child's occupancy (0 makes the left boundary
//     matter).
//   up: the upward carrier; cleared of underflow, possibly filled with
//     the child's new first key.
//   error: cleared, then set on failure.
//
// Returns: true on success; false with a named error on a pin failure
// (nothing written).
bool BPlusTree::borrow_from_right(std::uint32_t block, std::size_t child_index,
                                  std::uint32_t child_block,
                                  std::uint32_t right_block,
                                  bool child_is_leaf, std::size_t child_count,
                                  EraseUp& up, std::string& error) {
    // TODO(student): redistribute from the right sibling (slide 41).
    // Pin parent, child, sibling in ascending block order (pin_surgery).
    // Leaf child: move the sibling's first entry to the child's end and
    // rewrite the separator to the sibling's NEW smallest key ("copy up
    // 67 which is min key on remaining child"); an emptied child also
    // rewrites (or propagates) its left boundary. Separator child:
    // rotate — the sibling's first child and first key move through the
    // parent. Release everything dirty; clear up.underflow.
    (void)block;
    (void)child_index;
    (void)child_block;
    (void)right_block;
    (void)child_is_leaf;
    (void)child_count;
    (void)up;
    (void)error;
    return false;
}
// BPlusTree::merge_child — the merge (slides 42-43, 47): the
// underflowed child folds into its left sibling (preferred) or right
// sibling, the parent drops the separator and the child pointer, and
// the vanishing page joins the free list. The parent's own underflow is
// reported upward. Surgery pins (including the leaf chain's neighbour)
// in ascending block order, first.
//
// Parameters:
//   block: the parent's block id.
//   child_index: the underflowed child's routing position.
//   child_block: the child's block id.
//   left_block: the left sibling's block id (kNoPage when absent).
//   right_block: the right sibling's block id (kNoPage when absent).
//   child_is_leaf: which caste the surgery moves.
//   child_count: the child's occupancy (an emptied survivor's minimum
//     moves, so the parent separator follows).
//   up: the upward carrier; filled with the parent's post-merge
//     underflow and any first-key change.
//   error: cleared, then set on failure.
//
// Returns: true on success; false with a named error on a pin failure
// (nothing written).
bool BPlusTree::merge_child(std::uint32_t block, std::size_t child_index,
                            std::uint32_t child_block,
                            std::uint32_t left_block,
                            std::uint32_t right_block, bool child_is_leaf,
                            std::size_t child_count, EraseUp& up,
                            std::string& error) {
    // TODO(student): the merge (slides 42-43, 47). Pin parent,
    // survivor, vanished, and (leaf merges) the vanished page's chain
    // neighbour in ascending block order (pin_surgery). Merge with the
    // left sibling when there is one, else the right: a leaf merge moves
    // every entry to the survivor's end and closes the chain around the
    // vanished page; a separator merge appends the boundary pair (the
    // vanished page's first child) and the vanished page's pairs. Drop
    // the separator from the parent (erase_pair), chain the vanished
    // page onto the free list (chain_freed), and report the parent's own
    // underflow upward (up.underflow).
    (void)block;
    (void)child_index;
    (void)child_block;
    (void)left_block;
    (void)right_block;
    (void)child_is_leaf;
    (void)child_count;
    (void)up;
    (void)error;
    return false;
}
// BPlusTree::rebuild_from_leaves — the crossed-subtree erase's simple
// correct repair: collect every entry by walking the leaf chain, close
// the file, remove it, and re-run build. The rare case (a run spanning
// a subtree boundary, with the erased leaf then underflowing or moving
// its first key) traded for simplicity — see the design's trade-off.
//
// Parameters:
//   error: cleared, then set on failure.
//
// Returns: true on success (the index is open and balanced again);
// false with a named error when the chain cannot be walked, the file
// cannot be reset, or the rebuild fails.
bool BPlusTree::rebuild_from_leaves(std::string& error) {
    std::vector<std::pair<tuple::Column, idx::Rid>> entries;
    std::uint32_t current = first_leaf_;
    while (current != idx::kNoPage) {
        std::size_t frame = 0;
        BptLeafPage leaf;
        if (!pin_leaf(current, leaf, frame, error)) {
            return false;
        }
        const std::size_t n = leaf.count();
        const std::uint32_t next = leaf.next_leaf();
        for (std::size_t i = 0; i < n; ++i) {
            const idx::DataEntry entry = leaf.get_entry(i);
            entries.emplace_back(codec_.decode(entry.key.data()), entry.rid);
        }
        release(manager_, frame, false);
        current = next;
    }
    if (!close(error)) {
        return false;
    }
    if (std::remove(path_.c_str()) != 0) {
        error = "cannot reset the index file for the rebuild";
        return false;
    }
    return build(path_, entries, error);
}

// --- verify ---------------------------------------------------------------------
// BPlusTree::verify — the full walk (design D1's verify contract): the
// tree (reachability, occupancy, order, separator exactness, leaf-depth
// uniformity, dangling rids), the free list (no cycles, no overlap),
// block coverage, the leaf chain (global order, total), and the header
// total. A clean tree reports no findings.
//
// Parameters:
//   occupied: the caller's rid judge — true when the rid resolves to an
//     occupied heap slot.
//   entry_count: receives the walked total.
//   findings: cleared, then filled with named diagnostics.
//   error: cleared, then set on failure.
//
// Returns: true when the walk completed (findings may be non-empty —
// those are the report); false with a named error when the index is
// closed or a pin fails mid-walk.
bool BPlusTree::verify(const std::function<bool(const idx::Rid&)>& occupied,
                       std::size_t& entry_count,
                       std::vector<std::string>& findings, std::string& error) {
    findings.clear();
    entry_count = 0;
    if (!open_) {
        error = "index is not open";
        return false;
    }
    VerifyWalk walk{occupied, findings, entry_count};
    const std::uint32_t allocated =
        last_block_ == idx::kNoPage ? 0 : last_block_;
    walk.seen.assign(static_cast<std::size_t>(allocated) + 1, 0);

    // The tree walk: structure, occupancy, order, exactness, depths.
    std::optional<std::string> tree_min;
    if (root_ != idx::kNoPage) {
        if (!verify_rec(root_, 1, true, walk, error, &tree_min)) {
            return false;
        }
    }

    // The free list: every freed block marked, no cycles, no overlap
    // with the live tree.
    std::uint32_t current = freed_head_;
    while (current != idx::kNoPage) {
        if (current >= walk.seen.size()) {
            findings.push_back("free list names block " +
                               std::to_string(current) +
                               " beyond the file");
            break;
        }
        if (walk.seen[current] != 0) {
            findings.push_back("free list reaches block " +
                               std::to_string(current) + " twice");
            break;
        }
        walk.seen[current] = 1;
        std::size_t frame = 0;
        if (!manager_.pin(file_, current, frame, error)) {
            return false;
        }
        const std::uint32_t next =
            read_u32(manager_.frame(frame).buffer, 0);
        release(manager_, frame, false);
        current = next;
    }

    // Coverage: every allocated block is the header, tree-reachable, or
    // freed.
    for (std::uint32_t block = 1; block <= allocated; ++block) {
        if (walk.seen[block] == 0) {
            findings.push_back("block " + std::to_string(block) +
                               " is reachable from nowhere");
        }
    }

    // The leaf chain: global order, total, and first-leaf agreement.
    std::size_t chain_total = 0;
    std::uint32_t chain_current = first_leaf_;
    const std::string* previous_last = nullptr;
    std::string previous_storage;
    std::vector<char> on_chain(walk.seen.size(), 0);
    while (chain_current != idx::kNoPage) {
        if (chain_current >= on_chain.size()) {
            findings.push_back("leaf chain names block " +
                               std::to_string(chain_current) +
                               " beyond the file");
            break;
        }
        if (on_chain[chain_current] != 0) {
            findings.push_back("leaf chain loops at block " +
                               std::to_string(chain_current));
            break;
        }
        on_chain[chain_current] = 1;
        std::size_t frame = 0;
        BptLeafPage leaf;
        if (!pin_leaf(chain_current, leaf, frame, error)) {
            return false;
        }
        const std::size_t n = leaf.count();
        for (std::size_t i = 0; i < n; ++i) {
            const idx::DataEntry entry = leaf.get_entry(i);
            if (i > 0) {
                const idx::DataEntry previous = leaf.get_entry(i - 1);
                if (codec_.compare(previous.key.data(), entry.key.data()) > 0) {
                    findings.push_back("leaf " + std::to_string(chain_current) +
                                       " is out of order at entry " +
                                       std::to_string(i));
                }
            }
        }
        if (previous_last != nullptr && n > 0) {
            if (codec_.compare(previous_last->data(),
                               leaf.get_entry(0).key.data()) > 0) {
                findings.push_back("the leaf chain's key order breaks at " +
                                   std::to_string(chain_current));
            }
        }
        if (n > 0) {
            previous_storage = leaf.get_entry(n - 1).key;
            previous_last = &previous_storage;
        }
        chain_total += n;
        const std::uint32_t next = leaf.next_leaf();
        release(manager_, frame, false);
        chain_current = next;
    }
    if (chain_total != walk.walked) {
        findings.push_back("the leaf chain's total (" +
                           std::to_string(chain_total) +
                           ") disagrees with the tree walk (" +
                           std::to_string(walk.walked) + ")");
    }
    if (walk.min_depth != walk.max_depth && walk.min_depth != 0) {
        findings.push_back("leaves sit at depths " +
                           std::to_string(walk.min_depth) + " and " +
                           std::to_string(walk.max_depth));
    }
    if (walk.walked != entry_count_) {
        findings.push_back("the walked total (" +
                           std::to_string(walk.walked) +
                           ") disagrees with the header (" +
                           std::to_string(entry_count_) + ")");
    }
    return error.empty();
}
// BPlusTree::verify_rec — one verify walk step: marks the page seen
// (twice is a finding), pins and validates it, and at a separator
// recurses into every child (each separator must equal its child
// subtree's minimum — exactness) while checking occupancy and order; at
// a leaf, checks order, judges every rid through `occupied`, and feeds
// the depth range. The subtree's smallest key travels back to the
// parent.
//
// Parameters:
//   block: the page to verify.
//   depth: this page's depth (1 at the root; leaves sit at levels_).
//   is_root: the root's occupancy exemption.
//   walk: the shared walk state (seen marks, findings, totals, depth
//     range).
//   error: cleared, then set on failure.
//   subtree_min: when non-null, receives the subtree's smallest key (or
//     nullopt for an empty subtree).
//
// Returns: true when the step completed (findings may have grown);
// false with a named error on a pin failure.
bool BPlusTree::verify_rec(std::uint32_t block, std::uint32_t depth,
                           bool is_root, VerifyWalk& walk, std::string& error,
                           std::optional<std::string>* subtree_min) {
    // TODO(student): one verify walk step. Mark the page seen (a second
    // visit is the "reachable twice" finding; a block past the file is
    // "beyond the file"). The level decides the caste (a separator while
    // depth < levels_): check occupancy against the order rule (the root
    // is exempt), separators ascending, then recurse into the first
    // child and every pair's child — each separator must EQUAL its
    // child's subtree minimum (exactness), and each child returns its
    // minimum upward. At a leaf: order, every rid judged through
    // occupied (dangling findings), walked += count, depth range
    // updated, and the subtree minimum reported (nullopt when empty).
    
    // Resize seen when needed
    if(block >= walk.seen.size()){
        walk.seen.resize(block + 1);
    }

    /*
    Uses the walk.seen vector to mark blocks as they are visited. 
    If a block is encountered a second time, it logs a "reachable twice" finding,
    preventing infinite recursion caused by corrupted pointer cycles*/
    if(walk.seen[block] != 0){
        walk.findings.push_back("Page " + std::to_string(block) + " is reachable twice (cycle detected).");
        return true;
    }

    walk.seen[block] = 1;

    // Node type based depth vs level
    bool is_internal = depth < levels_;

    // Retrieves the fixed byte width of encoded keys for the indexed column.
    size_t kc = codec_.width();

    if (is_internal){
        BptSeparatorPage sep_page;
        size_t frame = 0;

        // Pin the separator page 
        if(!pin_separator(block, sep_page, frame, error)){
            walk.findings.push_back("Block " + std::to_string(block) + " is beyond the file or failed to pin as separator.");
            return true;
        }
       
        std::string attach_error;
        if(!sep_page.attach(attach_error)){
            walk.findings.push_back("Internal node " + std::to_string(block) + " failed page attach: " + attach_error);
        }

        size_t key_count = sep_page.count();

        // Occupancy check. The root is excluded
        if(!is_root && (key_count < min_separator_keys() || key_count > max_separator_keys())){
            walk.findings.push_back("Internal node " + std::to_string(block) + " violates minimum/maximum separator occupancy.");
        }

        // Check for key order violation. All values to the left must be lesser than the right values.
        for(size_t i = 1; i < key_count; i++){
            const char* previous_k = sep_page.separator_key(i-1);
            const char* current_k = sep_page.separator_key(i);
            if(codec_.compare(previous_k, current_k)<0){
                walk.findings.push_back("Internal node " + std::to_string(block) + " has unsorted separator keys.");
                break;
            }
        }

        // Recurse into children to check for exactness. Each separator key must equal its child's subtree minimum.
        size_t child_count = key_count + 1;
        std::optional<std::string>  first_child_min;
        uint32_t first_child_block = sep_page.first_child();
        if (!verify_rec(first_child_block, depth + 1, false, walk, error, &first_child_min)){
            release(manager_, frame, false);
            return false;
        }

        // Pass the first child min upwards 
        if(subtree_min){
            *subtree_min = first_child_min;
        }
  

        // Recurse into the rest of the children to check for exactness. Each separator key must equal its child's subtree minimum.
        for(size_t i = 0; i < key_count; i++){
            std::optional<std::string>  child_min;
            uint32_t child_block = sep_page.child(i);
            if (!verify_rec(child_block, depth + 1, false, walk, error, &child_min)){
                release(manager_, frame, false);
                return false;
            }

            const char* separator_keys = sep_page.separator_key(i);
            std::string separator_keys_str(separator_keys, kc);

            // Separator exactness. Separator must equal its childs subtree minimum
            if(child_min.has_value() && separator_keys_str != *child_min){
                walk.findings.push_back("Separator key exactness violation at internal node " + 
                                         std::to_string(block) + ", index " + std::to_string(i) + 
                                        " (separator does not match child's subtree minimum).");
            }
            }
        // Release the separator page
        release(manager_, frame, false);
        return true;

        }
        
        // Leaf node verification
        else{
            
            BptLeafPage leaf_page;
            size_t frame = 0;

            // Pin the leaf
            if(!pin_leaf(block, leaf_page, frame, error)){
                walk.findings.push_back("Block " + std::to_string(block) + " is beyond the file or failed to pin as leaf.");
                return true;
            }

            std::string attach_error;
            if(!leaf_page.attach(attach_error)){
                walk.findings.push_back("Leaf node " + std::to_string(block) + " failed page attach: " + attach_error);
            }

            size_t entry_count = leaf_page.count();
            bool has_previous = false;
            std::string previous_encoded; 

            // Check for key order violation. All values to the left must be lesser than the right values.
            for(size_t i = 0; i < entry_count; i++){
                auto entry = leaf_page.get_entry(i);
                if(has_previous){
                    if(codec_.compare(entry.key.data(), previous_encoded.data()) < 0){
                        walk.findings.push_back("Leaf node " + std::to_string(block) + " has unsorted keys.");
                        break;
                    }
                }
                previous_encoded = entry.key;
                has_previous = true;
            }

            // Occupancy check for leaf. Excludes the root
            if(!is_root && (entry_count < min_leaf_entries() || entry_count > max_leaf_entries())){
                walk.findings.push_back("Leaf node " + std::to_string(block) + " violates leaf occupancy rules.");
            }

            // Check depth. When depth min and max is 0 root
            if(walk.min_depth == 0 && walk.max_depth == 0){
                walk.min_depth = depth;
                walk.max_depth = depth;
            }
            // Change to new min and max depth
            else{
                if(depth < walk.min_depth){
                    walk.min_depth = depth;
                }
                if(depth > walk.max_depth){
                    walk.max_depth= depth;
                }
            }
            
            // Check red in occupied callback. (dangling check)
            for(size_t i = 0; i < entry_count; i++){
                auto entry = leaf_page.get_entry(i);
                if(walk.occupied && !walk.occupied(entry.rid)){
                    walk.findings.push_back("Dangling RID found in leaf node " + std::to_string(block) + ".");
                }
            }

            // Update to the walked path
            walk.walked += entry_count;

            // Report upwards the min subtree
            if(subtree_min){
                if(entry_count > 0){
                    auto first_entry = leaf_page.get_entry(0);
                    *subtree_min = first_entry.key;
                }
                else{
                    *subtree_min = std::nullopt;
                }
            }

            // Release the leaf page
            release(manager_, frame, false);

            return true;

        }
        
}

}
