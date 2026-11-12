#include "exec/Executor.h"

#include "exec/Binder.h"
#include "exec/Evaluator.h"
#include "schema/TableSchemaJson.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstdio>
#include <optional>
#include <algorithm>

namespace exec {

namespace {

// The evaluator's two-valued logic: booleans are ints 0/1.
bool truthy_value(const Value& value) {
    const int* v = std::get_if<int>(&value);
    return v != nullptr && *v != 0;
}

// A bound expression's value as a storable column (the same conversion
// the index bounds use).
tuple::Column value_to_column(const Value& value) {
    if (const int* v = std::get_if<int>(&value)) {
        return tuple::Column(*v);
    }
    if (const double* v = std::get_if<double>(&value)) {
        return tuple::Column(*v);
    }
    return tuple::Column(std::get<std::string>(value));
}

}


// --- the index-type dispatch (design D6) -------------------------------------

bool Executor::parse_index_type(const std::string& type,
                                SessionIndex::Kind& kind,
                                std::string& error) {
    if (type == "ISAM") {
        kind = SessionIndex::Kind::Isam;
        return true;
    }
    if (type == "BTREE") {
        kind = SessionIndex::Kind::Btree;
        return true;
    }
    error = "unknown index type '" + type +
            "' (the engine supports ISAM and BTREE)";
    return false;
}

bool Executor::SessionIndex::open(const std::string& path,
                                  std::string& error) const {
    return kind == Kind::Btree ? btree->open(path, error)
                               : isam->open(path, error);
}

bool Executor::SessionIndex::build(
    const std::string& path,
    const std::vector<std::pair<tuple::Column, idx::Rid>>& entries,
    std::string& error) const {
    return kind == Kind::Btree ? btree->build(path, entries, error)
                               : isam->build(path, entries, error);
}

bool Executor::SessionIndex::close(std::string& error) const {
    return kind == Kind::Btree ? btree->close(error) : isam->close(error);
}

bool Executor::SessionIndex::lookup(const tuple::Column& key,
                                    std::vector<idx::Rid>& out,
                                    std::string& error) const {
    return kind == Kind::Btree ? btree->lookup(key, out, error)
                               : isam->lookup(key, out, error);
}

bool Executor::SessionIndex::range(const std::optional<isam::RangeBound>& lo,
                                   const std::optional<isam::RangeBound>& hi,
                                   std::vector<idx::Rid>& out,
                                   std::string& error) const {
    if (kind == Kind::Isam) {
        return isam->range(lo, hi, out, error);
    }
    // The two RangeBound forms are shape-identical; convert at the seam.
    std::optional<bptree::RangeBound> blo;
    std::optional<bptree::RangeBound> bhi;
    if (lo) {
        blo = bptree::RangeBound{lo->value, lo->inclusive};
    }
    if (hi) {
        bhi = bptree::RangeBound{hi->value, hi->inclusive};
    }
    return btree->range(blo, bhi, out, error);
}

bool Executor::SessionIndex::insert(const tuple::Column& key, idx::Rid rid,
                                    std::string& error) const {
    return kind == Kind::Btree ? btree->insert(key, rid, error)
                               : isam->insert(key, rid, error);
}

bool Executor::SessionIndex::erase(const tuple::Column& key, idx::Rid rid,
                                   std::string& error) const {
    return kind == Kind::Btree ? btree->erase(key, rid, error)
                               : isam->erase(key, rid, error);
}

bool Executor::SessionIndex::verify(
    const std::function<bool(const idx::Rid&)>& occupied,
    std::size_t& entry_count, std::vector<std::string>& findings,
    std::string& error) const {
    return kind == Kind::Btree ? btree->verify(occupied, entry_count,
                                               findings, error)
                               : isam->verify(occupied, entry_count,
                                              findings, error);
}

Executor::Executor(bufman::BufferManager& manager,
                   const std::string& catalog_directory)
    : manager_(manager), catalog_directory_(catalog_directory) {
    // The canonical homes of table data and catalogs: created on demand,
    // silently tolerated when they already exist (failures surface later,
    // named, at file open).
    ::mkdir("data", 0755);
    ::mkdir(catalog_directory.c_str(), 0755);
}

std::string Executor::data_path(const std::string& table_name) const {
    return "data/" + table_name + ".tbl";
}

std::string Executor::index_path(const std::string& data_file,
                                 const std::string& column_name) const {
    return data_file + "." + column_name + ".idx";
}

bool Executor::load_catalog(const std::string& name, std::string& error) {
    if (catalogs_.count(name) != 0) {
        return true;
    }
    CatalogEntry entry;
    const std::string path = catalog_directory_ + "/" + name + ".json";
    if (!schema::load_table_schema(path, entry.schema, entry.indexes, error)) {
        error = "unknown table '" + name + "' (" + error + ")";
        return false;
    }
    catalogs_.emplace(name, std::move(entry));
    return true;
}

bool Executor::save_catalog(const std::string& name, std::string& error) {
    const auto found = catalogs_.find(name);
    if (found == catalogs_.end()) {
        error = "table '" + name + "' is not loaded";
        return false;
    }
    const std::string path = catalog_directory_ + "/" + name + ".json";
    if (!schema::save_table_schema(path, found->second.schema,
                                   found->second.indexes, error)) {
        return false;
    }
    return true;
}

void Executor::forget_file(const std::string& path) {
    files_.erase(path);
}

const schema::TableSchema* Executor::table(
    const std::string& name, std::vector<schema::TableIndex>& indexes,
    std::string& error) {
    if (!load_catalog(name, error)) {
        return nullptr;
    }
    indexes = catalogs_.at(name).indexes;
    return &catalogs_.at(name).schema;
}

heapfile::HeapFile* Executor::file_for(const std::string& table_name,
                                       const std::string& path,
                                       const schema::TableSchema& schema,
                                       bool create_when_missing,
                                       std::string& error) {
    const auto found = files_.find(path);
    if (found != files_.end()) {
        if (found->second.table_name != table_name) {
            error = "'" + path + "' is bound to table '" +
                    found->second.table_name + "'";
            return nullptr;
        }
        return found->second.file.get();
    }
    auto file = std::make_unique<heapfile::HeapFile>(manager_);
    if (!file->open(path, schema, error)) {
        if (!create_when_missing || !file->create(path, schema, error) ||
            !file->open(path, schema, error)) {
            return nullptr;
        }
    }
    SessionFile entry;
    entry.table_name = table_name;
    entry.file = std::move(file);

    // Open every catalog-declared index for this data file, building
    // any whose file is missing (lazy rebuild, spec: missing index file
    // rebuilt from the data file's rows).
    std::vector<schema::TableIndex> catalog_indexes;
    if (!load_catalog(table_name, error)) {
        return nullptr;
    }
    catalog_indexes = catalogs_.at(table_name).indexes;
    for (const schema::TableIndex& declared : catalog_indexes) {
        const std::optional<std::size_t> column =
            schema.column_index(declared.column);
        if (!column) {
            continue;  // the catalog is checked on write; tolerate here
        }
        SessionIndex session_index;
        session_index.column_index = *column;
        session_index.index_name = declared.name;
        session_index.column_name = declared.column;
        std::string type_error;
        if (!parse_index_type(declared.type, session_index.kind, type_error)) {
            error = type_error;
            return nullptr;
        }
        if (session_index.kind == SessionIndex::Kind::Btree) {
            session_index.btree =
                std::make_unique<bptree::BPlusTree>(manager_, schema, *column);
        } else {
            session_index.isam =
                std::make_unique<isam::IsamIndex>(manager_, schema, *column);
        }
        const std::string idx_path =
            index_path(path, declared.column);
        // A MISSING index file is a lazy rebuild (the missing-file rule);
        // an EXISTING file must open as the declared kind — a mismatch is
        // a named refusal, never a silent rebuild.
        struct stat file_stat;
        const bool index_file_exists =
            ::stat(idx_path.c_str(), &file_stat) == 0 && file_stat.st_size > 0;
        if (index_file_exists && !session_index.open(idx_path, error)) {
            return nullptr;  // named: wrong kind, wrong table, corruption
        }
        if (!index_file_exists) {
            error.clear();
            // Build from the file's current rows.
            std::vector<std::pair<tuple::Column, isam::Rid>> entries;
            heapfile::HeapFileIterator it = entry.file->scan();
            tuple::Tuple tuple;
            std::string scan_error;
            while (it.next(tuple, scan_error)) {
                entries.emplace_back(tuple.column(*column),
                                     isam::Rid{it.row_id().page_id,
                                               it.row_id().slot});
            }
            if (!scan_error.empty()) {
                error = scan_error;
                return nullptr;
            }
            if (!session_index.build(idx_path, entries, error)) {
                return nullptr;
            }
        }
        entry.indexes.push_back(std::move(session_index));
    }

    const auto inserted =
        files_.emplace(path, std::move(entry)).first;
    return inserted->second.file.get();
}

bool Executor::close_all(std::string& error) {
    error.clear();
    bool ok = true;
    for (auto& [path, entry] : files_) {
        std::string close_error;
        // The session's open index files flush and close first: their
        // dirty pages live in the shared pool as much as the data
        // file's do.
        for (SessionIndex& session_index : entry.indexes) {
            if (!session_index.close(close_error)) {
                if (!error.empty()) {
                    error += "; ";
                }
                error += close_error;
                ok = false;
            }
        }
        if (!entry.file->close(close_error)) {
            if (!error.empty()) {
                error += "; ";
            }
            error += close_error;
            ok = false;
        }
    }
    files_.clear();
    return ok;
}

bool Executor::insert_into(const std::string& table_name,
                           const std::string& path,
                           const tuple::Tuple& tuple,
                           heapfile::RowId& row_id, std::string& error) {
    if (!load_catalog(table_name, error)) {
        return false;
    }
    heapfile::HeapFile* file =
        file_for(table_name, path, catalogs_.at(table_name).schema, true,
                 error);
    if (file == nullptr) {
        return false;
    }
    if (!file->insert(tuple, row_id, error)) {
        return false;
    }
    const auto found = files_.find(path);
    for (const SessionIndex& session_index : found->second.indexes) {
        if (!session_index.insert(
                tuple.column(session_index.column_index),
                isam::Rid{row_id.page_id, row_id.slot}, error)) {
            return false;
        }
    }
    return true;
}

bool Executor::count_rows(const std::string& table_name,
                          const std::string& path, std::size_t& count,
                          std::string& error, bool create_when_missing) {
    if (!load_catalog(table_name, error)) {
        return false;
    }
    heapfile::HeapFile* file =
        file_for(table_name, path, catalogs_.at(table_name).schema,
                 create_when_missing, error);
    if (file == nullptr) {
        return false;
    }
    count = 0;
    heapfile::HeapFileIterator it = file->scan();
    tuple::Tuple tuple;
    std::string scan_error;
    while (it.next(tuple, scan_error)) {
        ++count;
    }
    if (!scan_error.empty()) {
        error = scan_error;
        return false;
    }
    return true;
}

bool Executor::check_indexes(const std::string& table_name,
                             const std::string& path,
                             std::vector<std::string>& report, bool& clean,
                             std::string& error) {
    report.clear();
    clean = true;
    if (!load_catalog(table_name, error)) {
        return false;
    }
    const CatalogEntry& entry = catalogs_.at(table_name);
    heapfile::HeapFile* file =
        file_for(table_name, path, entry.schema, false, error);
    if (file == nullptr) {
        return false;
    }
    const auto found = files_.find(path);
    if (entry.indexes.empty()) {
        report.push_back("table '" + table_name + "' declares no indexes");
        return true;
    }
    // A slot is occupied iff the heap can fetch its record.
    const auto occupied = [file](const isam::Rid& rid) {
        tuple::Tuple scratch;
        std::string find_error;
        return file->find(heapfile::RowId{rid.page_id, rid.slot}, scratch,
                          find_error);
    };
    for (const SessionIndex& session_index : found->second.indexes) {
        std::size_t walked = 0;
        std::vector<std::string> findings;
        if (!session_index.verify(occupied, walked, findings, error)) {
            return false;
        }
        std::string line = session_index.index_name + ": " +
                           std::to_string(walked) + " entries";
        if (findings.empty()) {
            line += ", OK";
        } else {
            clean = false;
            line += ", " + std::to_string(findings.size()) + " finding(s)";
            for (const std::string& finding : findings) {
                line += "\n  " + finding;
            }
        }
        report.push_back(line);
    }
    return true;
}

// --- dispatch ----------------------------------------------------------------

bool Executor::execute(const sql::Statement& statement,
                       StatementResult& result, std::string& error) {
    result = StatementResult{};
    switch (statement.kind) {
        case sql::Statement::Kind::CreateTable:
            return handle_create_table(
                static_cast<const sql::CreateTableStatement&>(statement),
                result, error);
        case sql::Statement::Kind::DropTable:
            return handle_drop_table(
                static_cast<const sql::DropTableStatement&>(statement),
                result, error);
        case sql::Statement::Kind::CreateIndex:
            return handle_create_index(
                static_cast<const sql::CreateIndexStatement&>(statement),
                result, error);
        case sql::Statement::Kind::DropIndex:
            return handle_drop_index(
                static_cast<const sql::DropIndexStatement&>(statement),
                result, error);
        case sql::Statement::Kind::Insert:
            return handle_insert(
                static_cast<const sql::InsertStatement&>(statement), result,
                error);
        case sql::Statement::Kind::Select:
            return handle_select(
                static_cast<const sql::SelectStatement&>(statement), result,
                error);
        case sql::Statement::Kind::Update:
            return handle_update(
                static_cast<const sql::UpdateStatement&>(statement), result,
                error);
        case sql::Statement::Kind::Delete:
            return handle_delete(
                static_cast<const sql::DeleteStatement&>(statement), result,
                error);
    }
    error = "unknown statement kind";
    return false;
}

// --- DDL ---------------------------------------------------------------------

bool Executor::handle_create_table(
    const sql::CreateTableStatement& statement, StatementResult& result,
    std::string& error) {
    const std::string catalog_path =
        catalog_directory_ + "/" + statement.table + ".json";
    if (load_catalog(statement.table, error)) {
        error = "table '" + statement.table + "' already exists";
        return false;
    }
    error.clear();
    if (statement.columns.empty()) {
        error = "table '" + statement.table + "' declares no columns";
        return false;
    }
    // Build the catalog text from the column defs, then parse it through
    // the loader — the sanctioned writer of declared widths — and install
    // it through the writer. Duplicate/empty names are refused by parse.
    std::string text = "{\"table_name\":\"" + statement.table +
                       "\",\"columns\":[";
    for (std::size_t i = 0; i < statement.columns.size(); ++i) {
        const sql::ColumnDef& column = statement.columns[i];
        const char* type_text = "Integer";
        std::size_t size = 4;
        switch (column.type) {
            case sql::DataType::Integer:
                type_text = "Integer";
                size = 4;
                break;
            case sql::DataType::Double:
                type_text = "Double";
                size = 8;
                break;
            case sql::DataType::String:
                type_text = "String";
                size = column.size;
                break;
        }
        if (i > 0) {
            text += ",";
        }
        text += "{\"column_name\":\"" + column.name +
                "\",\"column_type\":\"" + type_text +
                "\",\"size\":" + std::to_string(size) + "}";
    }
    text += "],\"indexes\":[]}";
    schema::TableSchema schema("");
    std::vector<schema::TableIndex> no_indexes;
    if (!schema::parse_table_schema(text, schema, no_indexes, error)) {
        return false;
    }
    // The catalog writer is the DDL seam: one atomic install.
    if (!schema::save_table_schema(catalog_path, schema, {}, error)) {
        return false;
    }
    CatalogEntry entry;
    entry.schema = std::move(schema);
    catalogs_.emplace(statement.table, std::move(entry));
    result.message = "table '" + statement.table + "' created";
    return true;
}

bool Executor::handle_drop_table(const sql::DropTableStatement& statement,
                                 StatementResult& result, std::string& error) {
    if (!load_catalog(statement.table, error)) {
        return false;  // unknown table: named error from the loader
    }
    const std::string data_file = data_path(statement.table);
    // Close the session's own handles first — the data file and every
    // index file the session opened — then erase from disk (erase_file
    // refuses while a file is open through the manager).
    const auto found = files_.find(data_file);
    if (found != files_.end()) {
        std::string close_error;
        for (const SessionIndex& session_index : found->second.indexes) {
            if (!session_index.close(close_error)) {
                error = close_error;
                return false;
            }
        }
        if (!found->second.file->close(close_error)) {
            error = close_error;
            return false;
        }
        files_.erase(found);
    }
    // Erase the data file and every declared index file. Missing files
    // are not errors — a table with no inserts has no data file.
    std::vector<schema::TableIndex> indexes =
        catalogs_.at(statement.table).indexes;
    if (std::remove(data_file.c_str()) != 0 && errno != ENOENT) {
        error = "cannot erase data file '" + data_file + "'";
        return false;
    }
    for (const schema::TableIndex& declared : indexes) {
        const std::string idx =
            index_path(data_file, declared.column);
        if (std::remove(idx.c_str()) != 0 && errno != ENOENT) {
            error = "cannot erase index file '" + idx + "'";
            return false;
        }
    }
    // Remove the catalog file itself, then the cache entry.
    const std::string catalog_path =
        catalog_directory_ + "/" + statement.table + ".json";
    if (std::remove(catalog_path.c_str()) != 0 && errno != ENOENT) {
        error = "cannot erase catalog '" + catalog_path + "'";
        return false;
    }
    catalogs_.erase(statement.table);
    result.message = "table '" + statement.table + "' dropped";
    return true;
}

bool Executor::handle_create_index(
    const sql::CreateIndexStatement& statement, StatementResult& result,
    std::string& error) {
    if (!load_catalog(statement.table, error)) {
        return false;
    }
    CatalogEntry& entry = catalogs_.at(statement.table);
    const std::optional<std::size_t> column =
        entry.schema.column_index(statement.column);
    if (!column) {
        error = "unknown column '" + statement.column + "' in table '" +
                statement.table + "'";
        return false;
    }
    for (const schema::TableIndex& declared : entry.indexes) {
        if (declared.column == statement.column) {
            error = "column '" + statement.column +
                    "' already carries an index (one index per column)";
            return false;
        }
        if (declared.name == statement.index_name) {
            error = "index '" + statement.index_name + "' already exists";
            return false;
        }
    }
    // The declared type: the USING word (uppercased) or the ISAM default.
    std::string wanted_type = statement.index_type;
    std::transform(wanted_type.begin(), wanted_type.end(),
                   wanted_type.begin(), ::toupper);
    if (wanted_type.empty()) {
        wanted_type = "ISAM";
    }
    SessionIndex::Kind wanted_kind = SessionIndex::Kind::Isam;
    std::string type_error;
    if (!parse_index_type(wanted_type, wanted_kind, type_error)) {
        error = type_error;
        return false;
    }
    // Record first (catalog write is all-or-nothing), then build.
    schema::TableIndex declared;
    declared.name = statement.index_name;
    declared.column = statement.column;
    declared.type = wanted_type;
    entry.indexes.push_back(declared);
    if (!save_catalog(statement.table, error)) {
        entry.indexes.pop_back();
        return false;
    }
    // If the data file is already open in this session, open the index
    // into it now (otherwise file_for opens it lazily on next use).
    const std::string data_file = data_path(statement.table);
    const auto found = files_.find(data_file);
    if (found != files_.end()) {
        SessionIndex session_index;
        session_index.column_index = *column;
        session_index.index_name = statement.index_name;
        session_index.column_name = statement.column;
        session_index.kind = wanted_kind;
        if (wanted_kind == SessionIndex::Kind::Btree) {
            session_index.btree =
                std::make_unique<bptree::BPlusTree>(manager_, entry.schema,
                                                    *column);
        } else {
            session_index.isam =
                std::make_unique<isam::IsamIndex>(manager_, entry.schema,
                                                  *column);
        }
        const std::string idx_path = index_path(data_file, statement.column);
        std::vector<std::pair<tuple::Column, isam::Rid>> entries;
        heapfile::HeapFileIterator it = found->second.file->scan();
        tuple::Tuple tuple;
        std::string scan_error;
        while (it.next(tuple, scan_error)) {
            entries.emplace_back(tuple.column(*column),
                                 isam::Rid{it.row_id().page_id,
                                           it.row_id().slot});
        }
        if (!scan_error.empty() ||
            !session_index.build(idx_path, entries, error)) {
            entry.indexes.pop_back();
            save_catalog(statement.table, error);
            if (!scan_error.empty()) {
                error = scan_error;
            }
            return false;
        }
        found->second.indexes.push_back(std::move(session_index));
    }
    result.message = "index '" + statement.index_name + "' created on '" +
                     statement.table + "(" + statement.column + ")'";
    return true;
}

bool Executor::handle_drop_index(const sql::DropIndexStatement& statement,
                                 StatementResult& result, std::string& error) {
    // The index name is unique per table; search every catalog in the
    // directory (loading the not-yet-loaded ones) so a DROP works even
    // in a session that never mentioned the table before.
    for (auto& [name, entry] : catalogs_) {
        for (std::size_t i = 0; i < entry.indexes.size(); ++i) {
            if (entry.indexes[i].name == statement.index_name) {
                return drop_index_in(name, entry, i, result, error);
            }
        }
    }
    // Load every catalog file in the directory and search again.
    std::vector<std::string> names;
    if (DIR* dir = opendir(catalog_directory_.c_str())) {
        while (const dirent* item = readdir(dir)) {
            const std::string file_name = item->d_name;
            const std::string suffix = ".json";
            if (file_name.size() > suffix.size() &&
                file_name.compare(file_name.size() - suffix.size(),
                                  suffix.size(), suffix) == 0) {
                names.push_back(
                    file_name.substr(0, file_name.size() - suffix.size()));
            }
        }
        closedir(dir);
    }
    for (const std::string& name : names) {
        if (!load_catalog(name, error)) {
            error.clear();  // an unreadable catalog is not a DROP error
            continue;
        }
        CatalogEntry& entry = catalogs_.at(name);
        for (std::size_t i = 0; i < entry.indexes.size(); ++i) {
            if (entry.indexes[i].name == statement.index_name) {
                return drop_index_in(name, entry, i, result, error);
            }
        }
    }
    error = "unknown index '" + statement.index_name + "'";
    return false;
}

bool Executor::drop_index_in(const std::string& name, CatalogEntry& entry,
                             std::size_t position, StatementResult& result,
                             std::string& error) {
    const std::string index_name = entry.indexes[position].name;
    const std::string data_file = data_path(name);
    const std::string idx =
        index_path(data_file, entry.indexes[position].column);
    // Close the session's open handles so the index file is flushed and
    // unpinned before erasure: the index itself, then the data file (the
    // whole session file goes — its indexes re-open lazily).
    const auto found = files_.find(data_file);
    if (found != files_.end()) {
        std::string close_error;
        for (SessionIndex& session_index : found->second.indexes) {
            if (session_index.index_name == index_name) {
                if (!session_index.close(close_error)) {
                    error = close_error;
                    return false;
                }
            }
        }
        if (!found->second.file->close(close_error)) {
            error = close_error;
            return false;
        }
        files_.erase(found);
    }
    if (std::remove(idx.c_str()) != 0 && errno != ENOENT) {
        error = "cannot erase index file '" + idx + "'";
        return false;
    }
    entry.indexes.erase(entry.indexes.begin() +
                        static_cast<long>(position));
    if (!save_catalog(name, error)) {
        return false;
    }
    result.message = "index '" + index_name + "' dropped";
    return true;
}

// --- INSERT ------------------------------------------------------------------

bool Executor::handle_insert(const sql::InsertStatement& statement,
                             StatementResult& result, std::string& error) {
    if (!load_catalog(statement.table, error)) {
        return false;
    }
    const CatalogEntry& entry = catalogs_.at(statement.table);
    const schema::TableSchema& schema = entry.schema;

    // Validate every row before any page is pinned: an INSERT is
    // all-or-nothing per statement.
    std::vector<tuple::Tuple> tuples;
    for (std::size_t r = 0; r < statement.rows.size(); ++r) {
        const std::vector<sql::ExprPtr>& row = statement.rows[r];
        if (row.size() != schema.column_count()) {
            error = "row " + std::to_string(r + 1) + " has " +
                    std::to_string(row.size()) + " value(s), table '" +
                    statement.table + "' has " +
                    std::to_string(schema.column_count()) + " columns";
            return false;
        }
        tuple::Tuple tuple;
        for (std::size_t c = 0; c < row.size(); ++c) {
            const sql::Expr& cell = *row[c];
            if (cell.kind != sql::Expr::Kind::Literal) {
                error = "row " + std::to_string(r + 1) + ", column '" +
                        schema.column_info(c).column_name() +
                        "': only literal values are supported";
                return false;
            }
            const sql::LiteralValue& literal = cell.value;
            const schema::ColumnSchema& meta = schema.column_info(c);
            switch (meta.column_type()) {
                case tuple::ColumnType::Integer:
                    if (literal.tag != sql::LiteralValue::Tag::Int) {
                        error = "row " + std::to_string(r + 1) + ", column '" +
                                meta.column_name() +
                                "': value must be an integer";
                        return false;
                    }
                    tuple.add(static_cast<int>(literal.i));
                    break;
                case tuple::ColumnType::Double:
                    if (literal.tag == sql::LiteralValue::Tag::Double) {
                        tuple.add(literal.d);
                    } else if (literal.tag == sql::LiteralValue::Tag::Int) {
                        tuple.add(static_cast<double>(literal.i));  // widening
                    } else {
                        error = "row " + std::to_string(r + 1) + ", column '" +
                                meta.column_name() +
                                "': value must be a number";
                        return false;
                    }
                    break;
                case tuple::ColumnType::String:
                    if (literal.tag != sql::LiteralValue::Tag::String) {
                        error = "row " + std::to_string(r + 1) + ", column '" +
                                meta.column_name() +
                                "': value must be a string";
                        return false;
                    }
                    tuple.add(literal.s);
                    break;
            }
        }
        if (!schema.validate(tuple, error)) {
            return false;  // width/NUL rules surface here, named
        }
        tuples.push_back(std::move(tuple));
    }

    heapfile::HeapFile* file =
        file_for(statement.table, data_path(statement.table), schema, true,
                 error);
    if (file == nullptr) {
        return false;
    }
    // Every insert lands in the heap first, then in each index of the
    // file (the mutation order the proposal pins).
    for (const tuple::Tuple& tuple : tuples) {
        heapfile::RowId row_id;
        if (!file->insert(tuple, row_id, error)) {
            return false;
        }
        // Maintain this file's indexes; a failure surfaces by name.
        const auto found = files_.find(data_path(statement.table));
        for (const SessionIndex& session_index : found->second.indexes) {
            if (!session_index.insert(
                    tuple.column(session_index.column_index),
                    isam::Rid{row_id.page_id, row_id.slot}, error)) {
                return false;
            }
        }
    }
    result.message = "inserted " + std::to_string(tuples.size()) + " row(s)";
    return true;
}

// --- SELECT ------------------------------------------------------------------

bool Executor::handle_select(const sql::SelectStatement& statement,
                             StatementResult& result, std::string& error) {
    // The surface refusals: the grammar accepts these, the executor
    // does not — each names its feature. (LIMIT, UPDATE, and DELETE are
    // part of this engine's surface and run below.)
    if (!statement.joins.empty()) {
        error = "joins are not supported";
        return false;
    }
    if (!statement.group_by.empty()) {
        error = "GROUP BY is not supported";
        return false;
    }
    if (!statement.order_by.empty()) {
        error = "ORDER BY is not supported";
        return false;
    }

    if (!load_catalog(statement.from.name, error)) {
        return false;
    }
    const CatalogEntry& entry = catalogs_.at(statement.from.name);
    const schema::TableSchema& schema = entry.schema;

    // Projection: star or plain columns (expressions in the projection
    // are outside the surface).
    std::vector<std::size_t> projected;
    result.rows.columns.clear();
    for (const sql::SelectItem& item : statement.items) {
        if (item.star) {
            for (std::size_t i = 0; i < schema.column_count(); ++i) {
                projected.push_back(i);
                result.rows.columns.push_back(
                    schema.column_info(i).column_name());
            }
            continue;
        }
        if (item.expr->kind != sql::Expr::Kind::ColumnRef) {
            if (item.expr->kind == sql::Expr::Kind::Function) {
                error = "aggregate functions are not supported";
                return false;
            }
            error = "expressions in the projection are not supported "
                    "(plain columns only)";
            return false;
        }
        const sql::Expr& column_expr = *item.expr;
        if (!column_expr.table.empty()) {
            error = "table-qualified columns need a join; single-table "
                    "queries use bare names";
            return false;
        }
        const std::optional<std::size_t> index =
            schema.column_index(column_expr.column);
        if (!index) {
            error = "unknown column '" + column_expr.column + "' in table '" +
                    statement.from.name + "'";
            return false;
        }
        projected.push_back(*index);
        result.rows.columns.push_back(
            item.alias.empty() ? column_expr.column : item.alias);
    }

    // Bind the WHERE (bind-time type errors happen before any page pin).
    BoundExprPtr where;
    if (statement.where) {
        if (!bind_expr(*statement.where, schema, where, error)) {
            return false;
        }
    }

    // Plan the access path. In oracle mode the planner sees no indexes,
    // so every conjunct stays residual and the heap path runs — the
    // suppression must gate the planner, not patch its output.
    std::vector<CatalogIndex> planner_indexes;
    if (allow_index_) {
        for (const schema::TableIndex& declared : entry.indexes) {
            const std::optional<std::size_t> column =
                schema.column_index(declared.column);
            if (column) {
                planner_indexes.push_back(
                    CatalogIndex{declared.name, *column});
            }
        }
    }
    ScanPlan plan = plan_where(std::move(where), planner_indexes);

    result.rows.access_path =
        plan.use_index ? "index scan" : "heap scan";

    // The session file, opened on first use (missing data file: no rows
    // can exist, so the SELECT answers empty).
    const std::string data_file = data_path(statement.from.name);
    heapfile::HeapFile* file =
        file_for(statement.from.name, data_file, schema, false, error);
    if (file == nullptr) {
        if (!error.empty()) {
            return false;  // a named refusal (a kind mismatch, say)
        }
        error.clear();
        return true;  // no data file: no rows exist, the answer is empty
    }
    const auto found = files_.find(data_file);

    result.rows.access_path =
        plan.use_index ? "index scan" : "heap scan";

    // Collect the candidate rows: index scan yields row ids; the heap
    // path scans every record.
    std::vector<heapfile::RowId> candidates;
    if (plan.use_index) {
        // The declared index for the searched column, through the
        // dispatch (either type answers the same range call).
        SessionIndex* session = nullptr;
        for (SessionIndex& candidate : found->second.indexes) {
            if (candidate.column_index == plan.bounds.column_index) {
                session = &candidate;
                break;
            }
        }
        if (session == nullptr) {
            plan.use_index = false;  // tolerate and scan
            result.rows.access_path = "heap scan";
        } else if (!plan.bounds.possible()) {
            return true;  // conflicting bounds: the index answers nothing
        } else {
            std::optional<isam::RangeBound> lo;
            std::optional<isam::RangeBound> hi;
            if (plan.bounds.lo) {
                lo = isam::RangeBound{value_to_column(*plan.bounds.lo),
                                      plan.bounds.lo_inclusive};
            }
            if (plan.bounds.hi) {
                hi = isam::RangeBound{value_to_column(*plan.bounds.hi),
                                      plan.bounds.hi_inclusive};
            }
            std::vector<idx::Rid> rids;
            if (!session->range(lo, hi, rids, error)) {
                return false;
            }
            candidates.reserve(rids.size());
            for (const idx::Rid& rid : rids) {
                candidates.push_back(
                    heapfile::RowId{rid.page_id, rid.slot});
            }
        }
    }

    if (!plan.use_index) {
        heapfile::HeapFileIterator it = file->scan();
        tuple::Tuple tuple;
        std::string scan_error;
        while (it.next(tuple, scan_error)) {
            candidates.push_back(it.row_id());
        }
        if (!scan_error.empty()) {
            error = scan_error;
            return false;
        }
    }

    // Fetch, residual-filter, project. Candidates are fetched one at a
    // time; nothing is pinned across rows. LIMIT stops the projection
    // once enough rows are out (limit 0 yields none).
    for (const heapfile::RowId& row_id : candidates) {
        if (statement.has_limit &&
            result.rows.rows.size() >= statement.limit) {
            break;
        }
        tuple::Tuple tuple;
        if (!file->find(row_id, tuple, error)) {
            return false;
        }
        bool keep = true;
        for (const BoundExprPtr& conjunct : plan.residual) {
            Value value;
            if (!evaluate(*conjunct, tuple, value, error)) {
                return false;
            }
            if (const int* b = std::get_if<int>(&value); b == nullptr || *b == 0) {
                keep = false;
                break;
            }
        }
        if (!keep) {
            continue;
        }
        std::vector<Value> row;
        row.reserve(projected.size());
        for (std::size_t column_index : projected) {
            const tuple::Column& column = tuple.column(column_index);
            switch (column.column_type()) {
                case tuple::ColumnType::Integer:
                    row.push_back(
                        Value(std::get<int>(column.column_value())));
                    break;
                case tuple::ColumnType::Double:
                    row.push_back(
                        Value(std::get<double>(column.column_value())));
                    break;
                case tuple::ColumnType::String:
                    row.push_back(
                        Value(std::get<std::string>(column.column_value())));
                    break;
            }
        }
        result.rows.rows.push_back(std::move(row));
    }
    return true;
}

// --- UPDATE ------------------------------------------------------------------
//
// Collect-then-apply (the p2dbeng semantics, index-aware): scan once for
// every matching row with its current values, then overwrite each with a
// new row computed from that row's OLD values — so SET assignments never
// observe each other's results — and rewrite only the indexes whose
// column a row's value changed (erase the old entry, insert the new).
// A collection failure applies nothing.

bool Executor::handle_update(const sql::UpdateStatement& statement,
                             StatementResult& result, std::string& error) {
    if (!load_catalog(statement.table, error)) {
        return false;
    }
    const CatalogEntry& entry = catalogs_.at(statement.table);
    const schema::TableSchema& schema = entry.schema;

    // Bind every SET target and expression, and the WHERE, before any
    // page is pinned — the bind-time refusals cost no I/O.
    struct BoundAssignment {
        std::size_t index;
        BoundExprPtr value;
    };
    std::vector<BoundAssignment> assignments;
    for (const sql::SetClause& clause : statement.sets) {
        const std::optional<std::size_t> index =
            schema.column_index(clause.column);
        if (!index) {
            error = "unknown column '" + clause.column + "' in SET";
            return false;
        }
        BoundExprPtr value;
        if (!bind_expr(*clause.value, schema, value, error)) {
            return false;
        }
        assignments.push_back(BoundAssignment{*index, std::move(value)});
    }
    BoundExprPtr where;
    if (statement.where) {
        if (!bind_expr(*statement.where, schema, where, error)) {
            return false;
        }
    }

    heapfile::HeapFile* file =
        file_for(statement.table, data_path(statement.table), schema, false,
                 error);
    if (file == nullptr) {
        if (!error.empty()) {
            return false;  // a named refusal
        }
        error.clear();  // no data file: no rows exist, nothing to update
        result.message = "updated 0 row(s)";
        return true;
    }

    // Collect: scan for matches, keeping each row's id and old values.
    struct Match {
        heapfile::RowId row_id;
        tuple::Tuple old_values;
    };
    std::vector<Match> matches;
    {
        heapfile::HeapFileIterator it = file->scan();
        tuple::Tuple tuple;
        std::string scan_error;
        while (it.next(tuple, scan_error)) {
            bool keep = true;
            if (where) {
                Value value;
                if (!evaluate(*where, tuple, value, error)) {
                    return false;
                }
                keep = truthy_value(value);
            }
            if (keep) {
                matches.push_back(Match{it.row_id(), tuple});
            }
        }
        if (!scan_error.empty()) {
            error = scan_error;
            return false;
        }
    }

    // Apply: rebuild each row from its old values (the new row is the old
    // row with SET positions replaced by expressions evaluated against
    // the OLD row — every assignment sees the old values, so simultaneous
    // SET semantics like the swap come free), write it back, and maintain
    // the indexes whose column changed.
    std::size_t updated = 0;
    for (const Match& match : matches) {
        tuple::Tuple fresh;
        for (std::size_t i = 0; i < match.old_values.column_count(); ++i) {
            const BoundAssignment* assigned = nullptr;
            for (const BoundAssignment& assignment : assignments) {
                if (assignment.index == i) {
                    assigned = &assignment;
                    break;
                }
            }
            if (assigned == nullptr) {
                fresh.add_column(match.old_values.column(i));
                continue;
            }
            Value value;
            if (!evaluate(*assigned->value, match.old_values, value, error)) {
                return false;
            }
            fresh.add_column(value_to_column(value));
        }
        if (!schema.validate(fresh, error)) {
            return false;  // the new row must still fit the catalog
        }
        if (!file->update(match.row_id, fresh, error)) {
            return false;
        }
        const auto found = files_.find(data_path(statement.table));
        for (const SessionIndex& session_index : found->second.indexes) {
            const tuple::Column& old_column =
                match.old_values.column(session_index.column_index);
            const tuple::Column& new_column =
                fresh.column(session_index.column_index);
            if (old_column == new_column) {
                continue;  // this index's key did not change
            }
            const isam::Rid rid{match.row_id.page_id, match.row_id.slot};
            if (!session_index.erase(old_column, rid, error) ||
                !session_index.insert(new_column, rid, error)) {
                return false;
            }
        }
        ++updated;
    }
    result.message = "updated " + std::to_string(updated) + " row(s)";
    return true;
}

// --- DELETE ------------------------------------------------------------------
//
// Collect-then-erase with the same discipline: matches are collected
// first, then removed from the heap and from every declared index, so no
// lookup or range ever returns a deleted row again. A collection failure
// erases nothing.

bool Executor::handle_delete(const sql::DeleteStatement& statement,
                             StatementResult& result, std::string& error) {
    if (!load_catalog(statement.table, error)) {
        return false;
    }
    const CatalogEntry& entry = catalogs_.at(statement.table);
    const schema::TableSchema& schema = entry.schema;

    BoundExprPtr where;
    if (statement.where) {
        if (!bind_expr(*statement.where, schema, where, error)) {
            return false;
        }
    }

    heapfile::HeapFile* file =
        file_for(statement.table, data_path(statement.table), schema, false,
                 error);
    if (file == nullptr) {
        if (!error.empty()) {
            return false;  // a named refusal
        }
        error.clear();  // no data file: no rows exist, nothing to delete
        result.message = "deleted 0 row(s)";
        return true;
    }

    // Collect: scan for matches, keeping each row's id and values (the
    // values are the index keys the apply phase erases).
    std::vector<std::pair<heapfile::RowId, tuple::Tuple>> matches;
    {
        heapfile::HeapFileIterator it = file->scan();
        tuple::Tuple tuple;
        std::string scan_error;
        while (it.next(tuple, scan_error)) {
            bool keep = true;
            if (where) {
                Value value;
                if (!evaluate(*where, tuple, value, error)) {
                    return false;
                }
                keep = truthy_value(value);
            }
            if (keep) {
                matches.emplace_back(it.row_id(), tuple);
            }
        }
        if (!scan_error.empty()) {
            error = scan_error;
            return false;
        }
    }

    std::size_t deleted = 0;
    for (const auto& [row_id, tuple] : matches) {
        if (!file->erase(row_id, error)) {
            return false;
        }
        const auto found = files_.find(data_path(statement.table));
        for (const SessionIndex& session_index : found->second.indexes) {
            const isam::Rid rid{row_id.page_id, row_id.slot};
            if (!session_index.erase(
                    tuple.column(session_index.column_index), rid, error)) {
                return false;
            }
        }
        ++deleted;
    }
    result.message = "deleted " + std::to_string(deleted) + " row(s)";
    return true;
}

}
